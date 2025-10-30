import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(project_root, 'src'))

from simple_hnsw import hnsw
import numpy as np

import time

train_data = np.random.rand(10000, 100)
query_data = np.random.rand(10, 100)

max_elements = 10000
dim = 100
M = 16
ef_construction = 32

# ==============
start_time = time.time()
# ==============

index = hnsw.HNSW('l2', dim)
index.init_index(max_elements, M, ef_construction)

index.insert_items(train_data)

# ==============
start_search_time = time.time()
# ==============

W = index.knn_search(query_data[0], 20)

# ==============
end_time = time.time()
# ==============

dist = [hnsw.l2_distance(train_data[w], query_data[0])**2 for w in W]

W = np.array(W)
dist = np.array(dist)

print("Simple HNSW:")
print("Time elapsed:", end_time - start_time)
print("Search time:", end_time - start_search_time)
print(W)
print(dist)

print("="*20)

import hnswlib

# ==============
start_time = time.time()
# ==============

index = hnswlib.Index('l2', dim)
index.init_index(max_elements, M, ef_construction)

index.add_items(train_data)

# ==============
start_search_time = time.time()
# ==============

W, dist = index.knn_query(query_data[0], 20)

# ==============
end_time = time.time()
# ==============

print("hnswlib:")
print("Time elapsed:", end_time - start_time)
print("Search time:", end_time - start_search_time)
print(W[0])
print(dist[0])