import numpy as np
import heapq

from typing import Literal
from numpy.typing import ArrayLike, NDArray

from .distance_metrics import l2_distance, cosine_distance

class HNSW:
    """
    A simple version of HNSW Algorithm
    """
    def __init__(self, space: Literal['l2', 'cosine'], dim: int) -> None:
        self.dim = dim
        if space == 'l2':
            self.distance = l2_distance
        else:
            self.distance = cosine_distance

    def init_index(self,
                   max_elements: int,
                   M: int = 16,
                   ef_construction: int = 200,
                   random_seed: int = 12345) -> None:
        self.max_elements = max_elements
        self.M = M
        self.maxM = M
        self.maxM0 = M * 2
        self.m_L = 1 / np.log(M)
        self.ef_construction = max(ef_construction, M)
        self.ef = 10

        self.graph: list[dict[int, dict]] = [] # graph[level][i] is a dict of (neighbor_id, dist)
        self.data: list[NDArray] = [] # data[i] is a embeded vector

        self.cur_element_count = 0
        self.entry_point = -1
        self.max_level = -1

        self.rng = np.random.default_rng(random_seed)

    def insert_items(self, data: ArrayLike) -> None:
        data = np.atleast_2d(data)
        for d in data:
            self.insert(d)

    def insert(self, q: ArrayLike) -> None:
        q = np.array(q)

        W = []
        ep = self.entry_point
        L = self.max_level
        l = int(-np.log(self.rng.uniform(0.0, 1.0)) * self.m_L)

        self.data.append(q)
        idx = self.cur_element_count
        self.cur_element_count += 1

        if self.entry_point != -1:
            for level in range(L, l, -1):
                W = self.seach_layer(q, ep, 1, level)
                ep = W[0]

            for level in range(min(L, l), -1, -1):
                W = self.seach_layer(q, ep, self.ef_construction, level)
                neighbors, dists = self.select_neighbors(q, W, self.M, level)

                # add bidirectional connections from neighbors to q
                for neighbor, dist in zip(neighbors, dists):
                    if idx not in self.graph[level]:
                        self.graph[level][idx] = {}
                    self.graph[level][idx][neighbor] = dist
                    self.graph[level][neighbor][idx] = dist

                maxM = self.maxM if level > 0 else self.maxM0
                for neighbor in neighbors:
                    neighbor_connections = list(self.graph[level][neighbor].keys())

                    if len(neighbor_connections) > maxM: # shrink connections of neighbor
                        neighbor_new_connections, neighbor_new_dist = self.select_neighbors(self.data[neighbor], neighbor_connections, maxM, level)

                        # set new neighborhood of neighbor
                        self.graph[level][neighbor].clear()
                        for n, d in zip(neighbor_new_connections, neighbor_new_dist):
                            self.graph[level][neighbor][n] = d

                ep = W[0]

        if L < l:
            self.entry_point = idx
            self.max_level = l
            for i in range(L, l):
                self.graph.append({idx: {}})

    def seach_layer(self,
                    q: ArrayLike,
                    ep: int,
                    ef: int,
                    level: int) -> list[int]:
        visited = set([ep])

        # These are containers of heap
        # each element is a tuple of (dist, id)
        candidates = [(self.distance(self.data[ep], q), ep)]
        W = [(-candidates[0][0], ep)] # negate first element to perform as max heap

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            f_dist, f_id = W[0]

            if c_dist > -f_dist:
                break

            for e_id, e_dist in self.graph[level][c_id].items():
                if e_id not in visited:
                    visited.add(e_id)
                    e_dist = self.distance(self.data[e_id], q)
                    # f_dist, f_id = W[0] # why the paper has this line?

                    if e_dist < -f_dist or len(W) < ef:
                        heapq.heappush(candidates, (e_dist, e_id))
                        heapq.heappush(W, (-e_dist, e_id))

                        if len(W) > ef:
                            heapq.heappop(W)

        nearest_neighbors = heapq.nlargest(ef, W)
        nearest_neighbors = [nn[1] for nn in nearest_neighbors]

        return nearest_neighbors

    def knn_search(self, q: ArrayLike, K: int = 1) -> list[int]:
        W = []
        ep = self.entry_point
        L = self.max_level

        for level in range(L, 0, -1):
            W = self.seach_layer(q, ep, 1, level)
            ep = W[0]

        W = self.seach_layer(q, ep, max(K, self.ef), 0)
        return W[:K]

    def select_neighbors(self,
                         q: ArrayLike,
                         W: list[int],
                         M: int,
                         level: int) -> tuple[list[int], list[float]]:
        proba = self.rng.uniform(0.0, 1.0)
        if proba <= 0.5:
            return self.select_neighbors_simple(q, W, M)
        return self.select_neighbors_heuristic(q, W, M, level)

    def select_neighbors_simple(self,
                                q: ArrayLike,
                                C: list[int],
                                M: int) -> tuple[list[int], list[float]]:
        nearest_neighbors = []

        for c in C:
            dist = -self.distance(self.data[c], q)
            heapq.heappush(nearest_neighbors, (dist, c))

            if len(nearest_neighbors) > M:
                heapq.heappop(nearest_neighbors)

        neighbors_id = []
        neighbors_distance = []

        for dist, id in nearest_neighbors:
            neighbors_id.append(id)
            neighbors_distance.append(-dist)

        return neighbors_id, neighbors_distance

    def select_neighbors_heuristic(self,
                                   q: ArrayLike,
                                   C: list[int],
                                   M: int,
                                   level: int,
                                   extend_candidates: bool = True,
                                   keep_pruned_connections: bool = True) -> tuple[list[int], list[float]]:
        R = []
        W = set(C)
        W_queue = []
        for c in C:
            heapq.heappush(W_queue, (self.distance(self.data[c], q), c))

        if extend_candidates:
            temp_W = W.copy()
            for e in temp_W:
                for en_id, en_dist in self.graph[level][e].items():
                    if en_id not in W:
                        W.add(en_id)
                        heapq.heappush(W_queue, (self.distance(self.data[en_id], q), en_id))

        W_d = []

        while W and len(R) < M:
            e_dist, e_id = heapq.heappop(W_queue)
            W.remove(e_id)

            if len(R) == 0 or e_dist < R[0][0]:
                heapq.heappush(R, (e_dist, e_id))
            else:
                heapq.heappush(W_d, (e_dist, e_id))

        if keep_pruned_connections:
            while W_d and len(R) < M:
                heapq.heappush(R, heapq.heappop(W_d))

        R_id = []
        R_dist = []

        while R:
            dist, id = heapq.heappop(R)
            R_id.append(id)
            R_dist.append(-dist)

        return R_id, R_dist

if __name__ == '__main__':
    train_data = np.random.rand(50, 100)
    query_data = np.random.rand(10, 100)

    max_elements = 50
    dim = 100
    M = 8
    ef_construction = 16

    index = HNSW('l2', dim)
    index.init_index(max_elements, M, ef_construction)

    index.insert_items(train_data)

    # for train in train_data:
    #     index.insert(train, M, 16, ef_construction, 1 / np.log(M))

    # for level in range(len(index.graph)):
    #     print('level:', level)
    #     for u, adj in index.graph[level].items():
    #         for v, d in adj.items():
    #             print(f"{u}, {v}, {d}")
    #         print()

    l = index.max_level
    ep = index.entry_point

    print(index.max_level)

    W = index.knn_search(query_data[0], 10)
    dists = [l2_distance(query_data[0], train_data[w])**2 for w in W]

    print(np.array(W))
    print(np.array(dists))

    print('='*10)
