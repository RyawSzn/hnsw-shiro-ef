import gzip
import json
import numpy as np
import h5py
from tqdm import tqdm
import glob
import hnswlib
from sklearn.datasets import make_blobs

# Dataset: MS MARCO V1

# wget https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-openai-ada2.tar -P collections/
# tar xvf collections/msmarco-passage-openai-ada2.tar -C collections/
print("Downloaded and extracted MS MARCO V1 dataset")
msmarco_v1_data_vecs = []
for i in tqdm(range(0, 89)):
    with gzip.open(str(i) + '.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            msmarco_v1_data_vecs.append(data['vector'])
msmarco_v1_data_vecs = np.array(msmarco_v1_data_vecs, dtype=np.float32)
print(msmarco_v1_data_vecs.shape)

msmarco_v2_query_vecs = []
with gzip.open('topics.msmarco-passage.dev-subset.openai-ada2.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        msmarco_v2_query_vecs.append(data['vector'])
msmarco_v2_query_vecs = np.array(msmarco_v2_query_vecs, dtype=np.float32)
print(msmarco_v2_query_vecs.shape)

# using hnswlib to compute ground truth
bf_msmarco_v1 = hnswlib.BFIndex(space='cosine', dim=msmarco_v1_data_vecs.shape[1])
bf_msmarco_v1.init_index(max_elements=msmarco_v1_data_vecs.shape[0])
bf_msmarco_v1.add_items(msmarco_v1_data_vecs)
gt_msmarco_v1, _ = bf_msmarco_v1.knn_query(msmarco_v2_query_vecs, k=1000)

with h5py.File("msmarco.hdf5", "w") as h5f:
    h5f.create_dataset("test", data=msmarco_v2_query_vecs)
    h5f.create_dataset("train", data=msmarco_v1_data_vecs)
    h5f.create_dataset("neighbors", data=gt_msmarco_v1)

print("Saved MS MARCO V1 dataset and ground truth to msmarco.hdf5")
! mv msmarco.hdf5 /home/ryawszn/experiments/shiro-ef/data/