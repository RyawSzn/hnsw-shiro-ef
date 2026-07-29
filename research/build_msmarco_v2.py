"""
MS MARCO V2.

$lux
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_00.npy
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_01.npy
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_02.npy
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_03.npy
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_04.npy
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_05.npy
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_06.npy
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_07.npy
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_08.npy
wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/passages_npy/msmarco_v2.1_doc_segmented_09.npy

wget https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/queries_jsonl/queries.jsonl.gz

"""

import gzip
import json
import os
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from tqdm import tqdm

import hnswlib

ms_marco_v2_queries = []
with gzip.open("query/queries.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        ms_marco_v2_queries.append(json.loads(line))
ms_marco_v2_query_vecs = [query["emb"] for query in ms_marco_v2_queries]
ms_marco_v2_query_vecs = np.array(ms_marco_v2_query_vecs).astype(np.float32)

ms_marco_v2_data = []
for e_path in sorted(glob.glob("data/*.npy")):
    ms_marco_v2_data.append(np.load(e_path))
ms_marco_v2_data_vecs = np.vstack(ms_marco_v2_data).astype(np.float32)

# using hnswlib to compute ground truth
bf_msmarco_v1 = hnswlib.BFIndex(space="cosine", dim=ms_marco_v2_data_vecs.shape[1])
bf_msmarco_v1.init_index(max_elements=ms_marco_v2_data_vecs.shape[0])
bf_msmarco_v1.add_items(ms_marco_v2_data_vecs)
gt_msmarco_v2, _ = bf_msmarco_v1.knn_query(ms_marco_v2_query_vecs, k=1000)

with h5py.File("cohere.hdf5", "w") as h5f:
    h5f.create_dataset("test", data=ms_marco_v2_query_vecs)
    h5f.create_dataset("train", data=ms_marco_v2_data_vecs)
    h5f.create_dataset("neighbors", data=gt_msmarco_v2)

print("Saved MS MARCO V2 dataset and ground truth to cohere.hdf5")
