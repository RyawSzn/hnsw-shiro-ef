import os
import struct

import faiss
import h5py
import numpy as np


def read_fvecs(filename):
    print(f"Reading {filename}")
    with open(filename, "rb") as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            return np.empty((0, 0), dtype=np.float32)
        d = struct.unpack("<i", dim_bytes)[0]

        # Read the rest of the file
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)

        vec_size = 4 + d * 4
        num_vecs = file_size // vec_size

        print(f"Found {num_vecs} vectors of dimension {d}")

        # Use memmap for memory efficiency
        x = np.memmap(filename, dtype="float32", mode="r")
        x = x.reshape(num_vecs, d + 1)

        # Return a copy to keep it in RAM and drop the leading dimension column
        return x[:, 1:].copy()


def compute_exact_knn(base, query, metric="euclidean", k=100):
    print(f"Computing exact top-{k} neighbors using {metric} distance...")
    d = base.shape[1]

    if metric == "angular":
        # For angular (cosine) distance, we normalize L2 and use Inner Product
        faiss.normalize_L2(base)
        faiss.normalize_L2(query)
        index = faiss.IndexFlatIP(d)
    else:
        # Default to standard Euclidean (L2) distance
        index = faiss.IndexFlatL2(d)

    index.add(base)
    distances, neighbors = index.search(query, k)
    return distances, neighbors


def create_hdf5(name, base_file, query_file, metric="euclidean"):
    base = read_fvecs(base_file)
    query = read_fvecs(query_file)

    distances, neighbors = compute_exact_knn(base, query, metric, k=100)

    out_file = (
        f"/home/ryawszn/experiments/shiro-ef/data/{name}-{base.shape[1]}-{metric}.hdf5"
    )
    print(f"Saving to {out_file}\n")

    with h5py.File(out_file, "w") as f:
        f.create_dataset("train", data=base)
        f.create_dataset("test", data=query)
        f.create_dataset("neighbors", data=neighbors.astype(np.int32))
        f.create_dataset("distances", data=distances)


if __name__ == "__main__":
    datasets = [
        # # GIST uses Euclidean (L2) distance
        # (
        #     "gist",
        #     "/home/ryawszn/experiments/shiro-ef/data/fvecs/gist_base.fvecs",
        #     "/home/ryawszn/experiments/shiro-ef/data/fvecs/gist_query.fvecs",
        #     "euclidean",
        # ),
        # # Word2Vec is used with Cosine/Angular distance
        # (
        #     "word2vec",
        #     "/home/ryawszn/experiments/shiro-ef/data/fvecs/word2vec_base.fvecs",
        #     "/home/ryawszn/experiments/shiro-ef/data/fvecs/word2vec_query.fvecs",
        #     "angular",
        # ),
        # # Tiny5M (TinyImages GIST features) uses Euclidean (L2) distance
        # (
        #     "tiny5m",
        #     "/home/ryawszn/experiments/shiro-ef/data/fvecs/tiny5m_base.fvecs",
        #     "/home/ryawszn/experiments/shiro-ef/data/fvecs/tiny5m_query.fvecs",
        #     "euclidean",
        # ),
        # # Sift10M uses Euclidean (L2) distance
        # (
        #     "sift10m",
        #     "/home/ryawszn/experiments/shiro-ef/data/fvecs/sift10m_base.fvecs",
        #     "/home/ryawszn/experiments/shiro-ef/data/fvecs/sift10m_query.fvecs",
        #     "euclidean",
        # ),
        # Msong uses Euclidean (L2) distance
        (
            "msong",
            "/home/ryawszn/experiments/shiro-ef/data/fvecs/msong_base.fvecs",
            "/home/ryawszn/experiments/shiro-ef/data/fvecs/msong_query.fvecs",
            "euclidean",
        ),
    ]

    for name, base, query, metric in datasets:
        create_hdf5(name, base, query, metric)
