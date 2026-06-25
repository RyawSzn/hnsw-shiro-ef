  RR vs RD: Complete Comparison

## 1. Statistics Summary

### glove-100-angular

| Metric | ef=50 | statics=1024 |
|--------|-------|-------------|
| **Time** | 197 μs | 4,457 μs |
| **Visited nodes** | 1,105 | 17,297 |
| **Expanded nodes** | 46 | 1,024 |
| **Revisited edges** | 141 | 9,294 |
| **RR corr(recall)** | **0.808** | **0.841** (+4.1%) |
| **RD corr(recall)** | 0.800 | 0.729 (-8.8%) |
| **Eff(RR)** | 0.00410 | 0.00019 (-95.4%) |
| **Eff(RD)** | 0.00406 | 0.00016 (-96.1%) |

### deep-image-96-angular

| Metric | ef=50 | statics=1024 |
|--------|-------|-------------|
| **Time** | 231 μs | 5,490 μs |
| **Visited nodes** | 919 | 13,541 |
| **Expanded nodes** | 43 | 1,024 |
| **Revisited edges** | 146 | 11,120 |
| **RR corr(recall)** | **0.745** | **0.800** (+7.4%) |
| **RD corr(recall)** | 0.608 | 0.699 (+15.0%) |
| **Eff(RR)** | 0.00322 | 0.00015 (-95.3%) |
| **Eff(RD)** | 0.00263 | 0.00013 (-95.1%) |

## 2. Key Findings

### RR vs RD: Which is better?

**RR always beats RD in correlation with recall:**

| Dataset | RR corr | RD corr | Gap |
|---------|---------|---------|-----|
| glove-100 (ef=50) | **0.808** | 0.800 | +0.008 |
| glove-100 (1024) | **0.841** | 0.729 | +0.112 |
| deep-image (ef=50) | **0.745** | 0.608 | +0.137 |
| deep-image (1024) | **0.800** | 0.699 | +0.101 |

**RR is consistently better.** The gap is wider on deep-image (0.137) than glove (0.008).

### Why RR beats RD

**RR (Revisit Rank)** = weighted revisit ratio using exponential decay:
```
R_v = Σ_{i∈revisit} exp(-γ·i/N) / Σ_{all} exp(-γ·i/N)
```
- Near edges (small rank i) contribute more weight
- Far edges contribute less
- Captures **where** revisits happen in the traversal order

**RD (Revisit Density)** = simple ratio:
```
RD = revisited_edges / expanded_nodes
```
- Treats all revisits equally
- Loses positional information
- Diluted when expanded nodes count is small (ef=50: 46 expanded)

### ef=50 vs statics=1024 for RR

| Gain from statics=1024 | Cost |
|------------------------|------|
| glove: +4.1% correlation | 22.6x slower |
| deep: +7.4% correlation | 23.8x slower |

**The gain is marginal compared to the cost.**

### Quantile Analysis (ef=50, glove-100)

| RR Bin | Mean RR | Mean Recall |
|--------|---------|-------------|
| 0 (lowest) | 0.305 | 0.233 |
| 1 | 0.342 | 0.243 |
| 2 | 0.368 | 0.358 |
| 3 | 0.396 | 0.429 |
| 4 | 0.425 | 0.514 |
| 5 | 0.468 | 0.642 |
| 6 | 0.510 | 0.727 |
| 7 | 0.574 | 0.834 |
| 8 | 0.647 | 0.929 |
| 9 (highest) | 0.726 | 0.970 |

**Monotonic relationship**: higher RR → higher recall. This is exactly what we want for the lookup table.

## 3. Recommendations

### For per-query estimation (inference time):
**Use ef=50 with RR.**
- 197-231 μs per query
- 0.808/0.745 correlation with recall
- 22x faster than statics=1024 for only 4-7% less correlation

### For training the lookup table (offline):
**Use statics=1024 with RR.**
- One-time cost, not per-query
- Better correlation means better-trained lookup table
- 0.841/0.800 vs 0.808/0.745

### Final verdict:
**RR > RD** in all cases. Use ef=50 for inference, statics=1024 for training.
