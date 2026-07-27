"""
Memory-efficient ground-truth builder for MS MARCO V1 (openai-ada2 embeddings).

$lux
wget --no-check-certificate -O- https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-openai-ada2.tar | tar -xv

$lux
git clone --depth 1 https://github.com/castorini/anserini-tools.git
ls anserini-tools/topics-and-qrels/ | grep -i openai-ada2

"""

import gzip
import json
import os
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from tqdm import tqdm

import hnswlib

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = "/mnt/d/RyawSzn_/Downloads/msmarco-passage-openai-ada2"
QUERIES_PATH = os.path.join(
    DATA_DIR, "topics.msmarco-passage.dev-subset.openai-ada2.jsonl.gz"
)
OUT_PATH = "/home/ryawszn/experiments/shiro-ef/data/msmarco.hdf5"
NUM_SHARDS = 89
TOP_K = 1000

CHUNK_SIZE = 50_000  # vectors read back from disk per chunk during GT computation
# must be >= TOP_K; raise if you have RAM headroom for more speed
DIM = 1536  # ada2 embedding dim
NUM_WORKERS = min(cpu_count() // 4, NUM_SHARDS)  # parallel shard-parsing workers
# note: must be an int -- cpu_count() / N
# (true division) returns a float and
# crashes Pool(); use // (floor division)
NUM_QUERY_THREADS = cpu_count() // 4  # threads hnswlib uses per knn_query call

assert CHUNK_SIZE >= TOP_K, "CHUNK_SIZE must be >= TOP_K"


# ----------------------------------------------------------------------------
# Step 1: parse queries (small, fine to hold fully in RAM)
# ----------------------------------------------------------------------------
def load_queries():
    vecs = []
    with gzip.open(QUERIES_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            vecs.append(data["vector"])
    return np.array(vecs, dtype=np.float32)


# ----------------------------------------------------------------------------
# Step 2: stream passages straight into an HDF5 dataset (never fully in RAM),
# parsing shards in parallel across CPU cores, writing in original order.
# ----------------------------------------------------------------------------
def _parse_shard(shard_idx):
    """CPU-bound: decompress + JSON-parse one shard fully, return as ndarray.
    Runs in a worker process so multiple shards parse concurrently.

    Parses each line directly into a preallocated float32 numpy array rather
    than building a Python list of lists first. A Python list of 1536 floats
    per row costs ~32 bytes/element (list + float object overhead) vs 4
    bytes/element in numpy -- for a ~99k-row shard that's the difference
    between a ~5.3GB peak (list + its numpy conversion coexisting briefly)
    and a ~0.6GB peak (the final array only). This is what makes it safe to
    run several of these in parallel without exceeding available RAM.
    """
    path = os.path.join(DATA_DIR, f"{shard_idx}.jsonl.gz")

    # first pass: count lines only (cheap, no JSON parsing) so we can
    # preallocate the exact-size array up front -- avoids any Python list
    # of vectors ever existing at all.
    n_lines = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for _ in f:
            n_lines += 1

    arr = np.empty((n_lines, DIM), dtype=np.float32)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            arr[i] = data["vector"]
    return shard_idx, arr


def stream_passages_to_hdf5(h5f):
    train_ds = h5f.create_dataset(
        "train",
        shape=(0, DIM),
        maxshape=(None, DIM),
        dtype=np.float32,
        chunks=(4096, DIM),
    )
    total = 0

    # imap (ordered) parses shards in parallel across NUM_WORKERS processes,
    # but yields results in *original shard order* (0, 1, 2, ...) regardless
    # of which order they actually finish in -- this preserves the exact row
    # ordering of the original single-threaded script, while still getting
    # full parallel speedup on the CPU-bound decompress+parse work. Only
    # completed-but-not-yet-yielded shards are buffered, which stays small
    # in practice since shard parse times are similar across workers.
    with Pool(NUM_WORKERS) as pool:
        for shard_idx, arr in tqdm(
            pool.imap(_parse_shard, range(NUM_SHARDS)),
            total=NUM_SHARDS,
            desc="parsing shards (parallel) + writing",
        ):
            n = arr.shape[0]
            train_ds.resize(total + n, axis=0)
            train_ds[total : total + n] = arr
            total += n

    print(f"Wrote {total} passage vectors to '{train_ds.name}'")
    return total


# ----------------------------------------------------------------------------
# Step 3: streaming brute-force cosine top-K using hnswlib.BFIndex per chunk
# (same AVX-accelerated, multithreaded exact search as the original script),
# merging a running top-K across chunks so the full train set never needs
# to be in one BFIndex/one array at once.
# ----------------------------------------------------------------------------
def compute_ground_truth(h5f, query_vecs, n_train):
    n_queries = query_vecs.shape[0]

    # cosine *distance* (smaller = more similar) -- same convention hnswlib
    # BFIndex uses, so merging is a simple "keep smallest K" per query.
    best_dist = np.full((n_queries, TOP_K), np.inf, dtype=np.float32)
    best_idx = np.full((n_queries, TOP_K), -1, dtype=np.int64)

    train_ds = h5f["train"]

    for start in tqdm(
        range(0, n_train, CHUNK_SIZE), desc="streaming brute-force GT (hnswlib)"
    ):
        end = min(start + CHUNK_SIZE, n_train)
        chunk = train_ds[start:end]  # only this slice is read from disk into RAM
        chunk_n = chunk.shape[0]

        # Build a small brute-force index over just this chunk -- identical
        # search semantics to the original script's single big BFIndex, just
        # scoped to a bounded slice of the data at a time.
        bf = hnswlib.BFIndex(space="cosine", dim=DIM)
        bf.init_index(max_elements=chunk_n)
        bf.add_items(chunk)

        k = min(TOP_K, chunk_n)
        labels, distances = bf.knn_query(query_vecs, k=k, num_threads=NUM_QUERY_THREADS)
        global_ids = (labels + start).astype(np.int64)

        # merge current best with this chunk's local top-k, keep smallest-K overall
        cand_dist = np.concatenate([best_dist, distances.astype(np.float32)], axis=1)
        cand_idx = np.concatenate([best_idx, global_ids], axis=1)

        part = np.argpartition(cand_dist, TOP_K - 1, axis=1)[:, :TOP_K]
        rows = np.arange(n_queries)[:, None]
        best_dist = cand_dist[rows, part]
        best_idx = cand_idx[rows, part]

    # final sort ascending by distance (nearest/most-similar first), matching
    # the ordering hnswlib.BFIndex.knn_query returns natively.
    order = np.argsort(best_dist, axis=1)
    rows = np.arange(n_queries)[:, None]
    best_idx = best_idx[rows, order]

    return best_idx  # neighbor indices, shape (n_queries, TOP_K)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("Loading query vectors...")
    query_vecs = load_queries()
    print(f"Queries shape: {query_vecs.shape}")

    with h5py.File(OUT_PATH, "w") as h5f:
        h5f.create_dataset("test", data=query_vecs)

        print("Streaming passage vectors to disk (bounded RAM, parallel parse)...")
        n_train = stream_passages_to_hdf5(h5f)

        print(
            "Computing ground truth via streaming hnswlib brute-force (bounded RAM)..."
        )
        neighbors = compute_ground_truth(h5f, query_vecs, n_train)

        h5f.create_dataset("neighbors", data=neighbors)

    print(f"Saved MS MARCO V1 dataset and ground truth to {OUT_PATH}")


if __name__ == "__main__":
    main()
