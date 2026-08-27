# SHIRO-EF: Architecture & Code Reference

> **Shiro EF** (探索因子) — a per-query adaptive `ef` predictor for HNSW that satisfies a strict recall target while minimizing distance computations.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Core Idea](#core-idea)
3. [System Architecture Overview](#system-architecture-overview)
4. [Phase 1 — Score Computation (`estimator.h`)](#phase-1--score-computation-estimatorh)
5. [Phase 2 — Search with Statistics Collection (`hnswalg.h`)](#phase-2--search-with-statistics-collection-hnswalgh)
6. [Phase 3 — Calibration (`shiro_ef.h`)](#phase-3--calibration-shiro_efh)
   - [RecallEstimator](#recallestimator)
   - [EfAdapter](#efadapter)
   - [Table-Filling Methods (0–3)](#table-filling-methods-03)
   - [Convergence Buckets](#convergence-buckets)
7. [Phase 4 — Online Lookup (`sketch.h`)](#phase-4--online-lookup-sketchh)
8. [Data Flow: End-to-End](#data-flow-end-to-end)
9. [Key Data Structures](#key-data-structures)
10. [File Reference](#file-reference)
11. [Experiment Configuration](#experiment-configuration)
12. [Building & Running](#building--running)
13. [Design Patterns](#design-patterns)
14. [Parameter Cheat Sheet](#parameter-cheat-sheet)

---

## The Problem

Standard HNSW uses a **single global** `ef` (exploration factor) for all queries. A large `ef` wastes compute on easy queries; a small `ef` fails to meet recall on hard ones. The ideal `ef` is per-query and depends on local dataset density around that query.

**Goal:** predict the minimum `ef` needed to meet a target recall `r*` (e.g., 0.95) for every individual query, while measuring recall across the whole workload.

---

## Core Idea

During the first few steps of the HNSW base-layer search, two cheap signals are measured:

| Signal | Name | What it captures |
|--------|------|-----------------|
| **Revisit Rank** (`r_v`, 0–100) | `score` / convergence score | How many early neighbor edges were **already visited** (a proxy for local density / "hard-ness") |
| **Coefficient of Variation** (`cv`, 0–100) | `cv_score` | Spread of distances in the current top-k heap — high cv = heterogeneous neighborhood = harder query |

These two scalars index into a pre-built 2-D lookup table that returns the predicted `ef` for that query. The table is built offline from a small calibration sample.

---

## System Architecture Overview

```
            ┌────────────────────────────────────────────────────┐
            │                  OFFLINE CALIBRATION               │
            │                                                    │
            │  Sample queries ──► RecallEstimator                │
            │      ↓ (score, actual_recall) per query            │
            │  EfAdapter.init()  ──► EfRecallTable               │
            │      ↓ (table-filling / smoothing)                 │
            │  EfAdapter.init_with_convergence_buckets()         │
            │      ↓ (per-cv-bucket tables, 2-D)                 │
            │  EfAdapter.serialize() ──► *.bin on disk           │
            └──────────────────────────┬─────────────────────────┘
                                       │
            ┌──────────────────────────▼─────────────────────────┐
            │                   ONLINE QUERY                     │
            │                                                    │
            │  adaptiveSearchKnn()                               │
            │    Phase 1 ── collect stats_length edge evals      │
            │    Phase 2 ── compute score (Revisit Rank r_v)     │
            │    Phase 3 ── compute cv (CoV of top-k distances)  │
            │    Phase 4 ── Sketch.estimate_ef2(cv, r_v) → ef    │
            │    Phase 5 ── resize heap to ef, continue search   │
            └────────────────────────────────────────────────────┘
```

---

## Phase 1 — Score Computation (`estimator.h`)

**File:** `hnswlib/estimator.h`  
**Class:** `hnswdis::ApproximatedScoreCalculator`

### What it does

Computes the **Revisit Rank** `r_v` ∈ [0, 100] from a vector of edge evaluations collected during the first `stats_length` steps of the HNSW search.

### Key parameters

| Parameter | Type | Role |
|-----------|------|------|
| `alpha` | `float` | Truncation ratio — only the top `⌊n×alpha⌋` edges (sorted ascending by distance) are considered. Filters noise from distant neighbors. |
| `gamma` | `float` | Exponential decay rate — controls how steeply the geometric weight series falls off. Higher gamma = more weight on the nearest few edges. |

### Algorithm (`calculate_score`)

```
1. Sort edges ascending by distance.
2. Take top_n = ⌊n × alpha⌋ edges (nearest).
3. factor = exp(-gamma / top_n)
4. Geometric series: w_0=factor, w_1=factor², …
5. r_v = 100 × Σ(w_i where edge_i was already visited) / Σw_i
```

A **high** `r_v` means many nearby neighbors were already visited → the query landed in a dense, well-explored region → search converges quickly → lower `ef` needed.  
A **low** `r_v` means mostly fresh neighbors → sparse or irregular local geometry → harder query → higher `ef` needed.

### Edge format

Each edge is `std::pair<float, bool>`: `{distance, is_visited}`. The `bool` is `true` if the neighbor was already in the visited list when evaluated.

---

## Phase 2 — Search with Statistics Collection (`hnswalg.h`)

**File:** `hnswlib/hnswalg.h`  
**Method:** `adaptiveSearchBaseLayerST<>()` (template, called by `adaptiveSearchKnn`)

### Modification to standard HNSW search

Standard HNSW search uses a fixed `ef` from the start. The Shiro-EF version:

1. **Sets `ef = ∞`** at the beginning, letting the search run unbounded.
2. Collects every edge evaluation into `edge_evals` (up to `stats_limit = stats_length`).
3. When `edge_evals` reaches `stats_limit`:
   - Calls `score_calculator.compute_score(...)` → gets `r_v` score.
   - Computes **CV** from the current top-k heap:
     ```
     cv = std(distances) / mean(distances)
     cv_score = clamp(cv × 400, 0, 100)
     ```
   - Calls `sketch->estimate_ef2(cv_score, r_v)` → predicted `ef`.
   - **Resizes** the heap to `ef`, sets the real `lowerBound`, and continues.
4. After `stats_limit` edges, switches to normal bounded HNSW behavior.

This ensures the overhead of statistics collection is bounded to the first `stats_length` evaluations regardless of `ef`.

### `adaptiveSearchKnn` vs `adaptiveSearchKnnTest`

| Method | Purpose |
|--------|---------|
| `adaptiveSearchKnn` | Production path. Returns `{result_queue, conv_score}`. |
| `adaptiveSearchKnnTest` | Debug/profiling path. Also outputs the predicted `ef` used per query. |

---

## Phase 3 — Calibration (`shiro_ef.h`)

**File:** `hnswlib/shiro_ef.h`

### RecallEstimator

Runs `adaptiveSearchKnn` on a calibration set at a given `ef`, then groups queries by their `cv_score` bin and computes per-bin recall statistics.

**Recall statistics tuple:** `(score, avg_recall, median, p25, p5, count)`

**`score_to_query_map`:** maps each `cv_score` integer → list of query indices that fell into that bin. Used later to iteratively raise `ef` for under-performing bins.

```
RecallEstimator(hnsw, data, query_vecs, ground_truth, k, score_cal, ef, stats_length, min_q)
    └── hnsw_search_rv_and_cv(...)           → (labels, r_v_score, cv)  per query
    └── compute_recall(ground_truth, labels) → recall per query
    └── group by cv_score bin
    └── filter bins with < min_queries_per_score samples
    └── compute (avg, median, p25, p5) per bin → recall_statistics
```

### EfAdapter

The main calibration engine. Builds a mapping `cv_score → ef` that satisfies `target_recall`.

#### `init()` — 1-D calibration (single convergence bucket)

```
1. Run RecallEstimator at ef=k       → first_recall_estimator
2. Run RecallEstimator at ef=1.5k    → second_recall_estimator
3. Build initial EfRecallTable:
     for each cv_score bin:
         ef_recall_list = [(k, recall_at_k), (1.5k, recall_at_1.5k)]
4. For each bin still below target_recall:
     iterative binary-search-like EF raising:
         ef += Δef × (gap_to_target / recall_diff)
         run searchKnn on just this bin's queries
         append (ef, recall) to ef_recall_list
         stop when recall ≥ target or ef == ef_upper_bound
5. Apply table-filling method (0/1/2/3) to smooth holes
6. Compute WAE (weighted average ef over all queries)
```

**WAE calculation modes (`WAE_CALC_METHOD`):**
- `0` — Use the minimum `ef` that achieved `target_recall` (most aggressive).
- `1` — Average of all `ef` values that achieved `target_recall` (more conservative).

### Table-Filling Methods (0–3)

These handle **sparse cv_score bins** — bins with too few calibration samples to produce reliable `ef` estimates.

| Method | Strategy |
|--------|---------|
| **0** | Keep only bins that were calibrated; no interpolation. Gaps in the table are empty. |
| **1** | Split scores into a **continuous block** (densely sampled) and discrete outlier tails. Fill tails with weighted averages clamped by the continuous block's average (`left ≥ cont_avg`, `right ≤ cont_avg`). |
| **2** | Same splitting as Method 1, but fills gaps inside the continuous block with **Inverse Distance Weighting (IDW)** interpolation (1/dist²). Extrapolates tails with clamped pivots. |
| **3** | Raw-points-only per bucket (no intra-bucket filling); the outer `init_with_convergence_buckets` then performs full **2-D Shepard (Mahalanobis-distance IDW)** interpolation across `(cv, score)` space, producing a dense 101×101 grid. |

### Convergence Buckets

`init_with_convergence_buckets()` partitions the calibration queries by their **convergence score** (`r_v`) into `n_convergence_buckets` groups:

```
queries sorted by r_v
   └── bucket 0: [0, chunk)         — hardest (lowest r_v)
   └── bucket 1: [chunk, 2×chunk)
   └── ...
   └── bucket n-1: [(n-1)×chunk, n) — easiest (highest r_v)
```

Key insight: **hard queries** (low r_v, bottom 20th percentile = `conv_p20`) are calibrated to `expected_recall` (e.g., 0.95), while **easy queries** are calibrated to a relaxed `easy_recall` (e.g., 0.98), since they're cheap and can afford extra accuracy.

Each bucket independently runs `init()`, producing its own `EfRecallTable`. At query time, the bucket nearest to the query's `r_v` is selected (with linear interpolation between adjacent buckets via `Sketch`).

**`n_convergence_buckets = 0`:** special mode — uses absolute `cv_score` bins rather than fractile-based bucketing. Each distinct integer `cv_score` gets its own bucket.

---

## Phase 4 — Online Lookup (`sketch.h`)

**File:** `hnswlib/sketch.h`  
**Class:** `hnswdis::Sketch`  
**Compile-time switch:** `SMOOTHING_METHOD` (0 = none, 1 = 3-point average)

### Purpose

Zero-overhead O(1) lookup: given `(cv_score, r_v)` → `ef`.

### `build_links(table)`

Pre-builds a 101-element `links[]` array that maps every integer score 0–100 to the nearest row index in the `EfRecallTable`. This avoids `lower_bound` binary search at query time.

### `lookup_ef(table, links, score)`

```
clamped = clamp(score, 0, 100)
index   = links[clamped]
for each (ef, recall) in table[index].second:
    if recall >= expected_recall: return ef
return table[index].second.back().ef   # safety fallback
```

### `smoothed_ef(score)` (SMOOTHING_METHOD=1)

```
return (ef(score-1) + ef(score) + ef(score+1)) / 3
```
Prevents cliff artifacts at bucket boundaries.

### `interpolate_conv_buckets(cv_score, conv)`

When convergence buckets are active:
1. Find the two adjacent bucket centers bracketing `conv`.
2. Linear-interpolate their `smoothed_ef` outputs.
3. Returns a blended `ef`.

### `estimate_ef2(cv_score, conv)` — the hot path

Called inside `adaptiveSearchBaseLayerST` every query:
```
if convergence_buckets:
    return interpolate_conv_buckets(cv_score, conv)
else:
    return smoothed_ef(ef_recall_table_single, links[0], cv_score)
```

---

## Data Flow: End-to-End

```
OFFLINE
═══════
datasets (HDF5) ──► load_index_and_data() ──► hnsw + query + gt

Sample ~2000 queries (LHS / random)
         │
         ▼
ApproximatedScoreCalculator(alpha, gamma)
         │
         ▼
RecallEstimator @ ef=k  ──────────────────────────────────────┐
RecallEstimator @ ef=1.5k                                     │
         │                                                    │
         ▼                                           EfRecallTable
EfAdapter.init_with_convergence_buckets()
  ├── compute r_v + cv for all calibration queries
  ├── compute conv_p20 (20th percentile of r_v)
  ├── partition into n_convergence_buckets groups
  ├── for each bucket: EfAdapter.init() → per-bucket EfRecallTable
  └── [FILLING_METHOD=3]: 2-D Shepard interpolation → 101×101 grid
         │
         ▼
EfAdapter.serialize() ──► estimation_table/*.bin

ONLINE
══════
EfAdapter.deserialize() ──► Sketch(tables, centers, expected_recall)
         │
         ▼
query_vector ──► adaptiveSearchKnn(k, stats_length, score_cal, &sketch)
  ├── Phase 1: multi-level greedy descent to base layer entry
  ├── Phase 2: BFS in base layer, ef=∞, collect edge_evals
  ├── Phase 3: at stats_limit edges:
  │     ├── compute r_v = score_cal.compute_score(edge_evals)
  │     ├── compute cv from top-k heap
  │     └── ef = sketch.estimate_ef2(cv_score, r_v)
  └── Phase 4: resize heap, continue with ef
         │
         ▼
top-k results + convergence_score
```

---

## Key Data Structures

### `EfRecallTable`
```cpp
// vector< pair<cv_score_int, vector<pair<ef, recall>>> >
// Outer: sorted by cv_score (0–100)
// Inner: monotonically increasing (ef, recall) pairs for that score bin
using EfRecallTable = std::vector<std::pair<int, std::vector<std::pair<int, float>>>>;
```

### `edge_evals`
```cpp
// Collected during search, passed to ApproximatedScoreCalculator
std::vector<std::pair<float, bool>>  // {distance, is_visited}
```

### `recall_statistics` (RecallEstimator)
```cpp
// tuple<cv_score, avg_recall, median, p25, p5, count>
std::vector<std::tuple<int, float, float, float, float, size_t>>
```

### `ExperimentConfig` (run.cpp)
```cpp
struct ExperimentConfig {
    std::string dataset, metric;
    size_t k;
    float alpha, gamma, expected_recall, easy_recall;
    int ef_upper_bound, repeat, sampling_size;
    int n_convergence_buckets, min_queries_per_score;
    size_t stats_length;
};
```

---

## File Reference

| File | Namespace / Class | Role |
|------|------------------|------|
| `hnswlib/hnswlib.h` | `hnswlib` | Base interfaces: `SpaceInterface`, `AlgorithmInterface`, `BaseSearchStopCondition`, `BaseFilterFunctor`, SIMD capability detection, POD serialization helpers |
| `hnswlib/estimator.h` | `hnswdis` | `ApproximatedScoreCalculator` — computes Revisit Rank `r_v` score from edge evaluations |
| `hnswlib/hnswalg.h` | `hnswlib` | `HierarchicalNSW<dist_t>` — full HNSW implementation including `adaptiveSearchKnn` and `adaptiveSearchBaseLayerST` |
| `hnswlib/shiro_ef.h` | `hnswdis` | `RecallEstimator`, `EfAdapter` — offline calibration pipeline; `compute_samplings`, serialization |
| `hnswlib/sketch.h` | `hnswdis` | `Sketch` — online O(1) lookup with smoothing and inter-bucket interpolation; `EfRecallTable` typedef |
| `hnswlib/space_l2.h` | `hnswlib` | L2 (Euclidean) distance with AVX/AVX512/SSE SIMD dispatch |
| `hnswlib/space_ip.h` | `hnswlib` | Inner product / cosine distance with SIMD dispatch |
| `hnswlib/bruteforce.h` | `hnswlib` | `BruteforceSearch<dist_t>` — exact kNN, used to compute ground truth |
| `hnswlib/visited_list_pool.h` | `hnswlib` | `VisitedListPool` — reusable visited-node bitsets for thread-safe search |
| `hnswlib/stop_condition.h` | `hnswlib` | `MultiVectorSearchStopCondition`, `EpsilonSearchStopCondition` |
| `experiments_driver/run.cpp` | — | Main benchmark driver; `g_experiments` config table; `online_exp()`, `train_convergence_buckets()`, `make_sketch()` |
| `experiments_driver/util.h` | `hnswdis` | HDF5 loading (`load_hdf5`), dataset normalization, index construction helpers, `load_index_and_data` |
| `research/` | Python | Visualization and ablation scripts (not part of the C++ runtime) |

---

## Experiment Configuration

All experiments are defined in the static array `g_experiments` in `experiments_driver/run.cpp`. Each entry is an `ExperimentConfig`:

```cpp
// dataset, metric, k, alpha, gamma, hard_recall, easy_recall, ef_ub, repeat, sample, n_cv_buckets, min_q, stats_len
{"sift10m-128-euclidean", "l2", 100, 1.0f, 16.0f, 0.95f, 0.98f, 1000, 5, 2000, 10, 3, 1 + 32 + 31*32}
//                                                                                         ↑
//                                                                              stats_length = 993 edges
//                                                                              (1 entry point + 32 neighbors
//                                                                               + 31 × 32 second-hop neighbors)
```

**`stats_length` formula:**  
`1 + M + (M-1) × M` — captures the entry point evaluation, all its neighbors, and the neighbors of all those neighbors (1.5-hop).  
For `M=16` (HNSW default for base layer `maxM0=32`): `1 + 32 + 31×32 = 993`.

### Compile-time switches

| Constant | File | Values | Effect |
|----------|------|--------|--------|
| `FILLING_METHOD` | `sketch.h` | 0/1/2/3 | How to fill sparse EfRecallTable bins |
| `WAE_CALC_METHOD` | `shiro_ef.h` | 0/1 | Min-ef vs average-ef WAE calculation |
| `WAE_METHOD` | `run.cpp` | 0/1 | Use sampling WAE (0) or reconstructed hard/easy WAE (1) |
| `SMOOTHING_METHOD` | `sketch.h` | 0/1 | No smoothing vs 3-point average |
| `GROUND_WAE` | `hnswalg.h` | 0/1 | When 1, never use ef < ef_baseline (ground truth WAE mode) |

---

## Building & Running

```bash
# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Setup dataset directories
export EXPERIMENTS_ROOT="/path/to/experiments"
./setup.sh    # creates data/ and index/

# First run (builds HNSW index from HDF5 data, then calibrates)
./build/run

# Capture logs
nohup ./build/run > output_shiro.log 2>&1 &
```

### Expected directory layout
```
$EXPERIMENTS_ROOT/
├── data/
│   ├── sift10m-128-euclidean.hdf5
│   └── ...
├── index/
│   └── sift10m-128-euclidean-M16-efc-500-parallel.hnsw
├── estimation_table/
│   └── sift10m-128-euclidean-ef_adapter--k100-ef.bin
└── sampling/
    └── sift10m-128-euclidean-samplings--k100-ef.bin
```

---

## Design Patterns

### 1. Two-phase search (bounded statistics collection)

The adaptive search first runs unbounded (`ef=∞`) for exactly `stats_length` edge evaluations, then pivots to a bounded search at the predicted `ef`. This decouples the prediction cost (always `O(stats_length)`) from the actual search cost.

### 2. Hard/easy query stratification

Queries are split at `conv_p20` (20th percentile of convergence scores). The bottom 20% are "hard" queries that get calibrated to `expected_recall`; the rest are "easy" and get calibrated to the more permissive `easy_recall`. This reduces the average `ef` for the easy majority.

### 3. Serializable calibration artifact

The entire `EfAdapter` (all bucket tables + convergence centers + metadata) serializes to a compact binary `.bin` file via `writeBinaryPOD`. The online path deserializes once at startup and queries purely in-memory arrays.

### 4. Score-bin guards (`min_queries_per_score`)

Score bins with fewer than `min_queries_per_score` calibration queries are dropped. This prevents rare outlier bins from producing statistically noisy `ef` estimates.

### 5. Table smoothing over `score` (Sketch)

The `Sketch` applies a 3-point moving average (`score-1, score, score+1`) when `SMOOTHING_METHOD=1`, preventing sharp `ef` cliffs at bin boundaries from causing recall violations on borderline queries.

### 6. `Sketch` vs `EfAdapter`

`EfAdapter` is the **offline** calibration object — heavy, owns all data, builds tables iteratively.  
`Sketch` is the **online** read-only wrapper — holds only raw pointers into the adapter's data, performs only index lookups and interpolation. Created once per experiment run via `make_sketch(adapter, recall)`.

---

## Parameter Cheat Sheet

| Parameter | Typical Value | Effect of Increasing |
|-----------|--------------|---------------------|
| `alpha` | 1.0 | More edges in Revisit Rank window → richer signal, slightly slower score |
| `gamma` | 16.0 | Heavier exponential decay → more weight on nearest few edges |
| `stats_length` | 993 (`1+32+31×32`) | Richer statistics window → better prediction, higher overhead before ef is set |
| `expected_recall` | 0.95 | Stricter target → higher `ef` for hard bins |
| `easy_recall` | 0.98 | Stricter easy queries → slightly higher average `ef` |
| `ef_upper_bound` | 1000–5000 | Higher safety cap → protects against tail latency |
| `n_convergence_buckets` | 10 | More buckets → finer r_v stratification, more calibration time |
| `sampling_size` | 2000 | More calibration queries → stabler bin statistics |
| `min_queries_per_score` | 3 | Higher → fewer but more reliable bins |
| `FILLING_METHOD` | 3 (IDW) | Method 3 is most complete but most expensive offline; Method 0 is fastest but has gaps |
| `M` (HNSW) | 16 | Higher → better graph quality, higher memory |
| `ef_construction` | 500 | Higher → better index quality, longer build time |
