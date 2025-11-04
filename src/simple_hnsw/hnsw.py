import numpy as np
import pandas as pd
import heapq

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        self.level: list[int] = [] # level of datas

        self.cur_element_count = 0
        self.entry_point = -1
        self.max_level = -1

        self.rng = np.random.default_rng(random_seed)

    def insert_items(self, data: ArrayLike, visualize: bool = False) -> None:
        data = np.atleast_2d(data)
        for d in data:
            self.insert(d, visualize)

    def insert(self, q: ArrayLike, visualize: bool = False) -> None:
        q = np.array(q)

        W = []
        ep = self.entry_point
        L = self.max_level
        l = int(-np.log(self.rng.uniform(0.0, 1.0)) * self.m_L)

        self.data.append(q)
        self.level.append(l)
        idx = self.cur_element_count
        self.cur_element_count += 1

        if self.entry_point != -1:
            for level in range(L, l, -1):
                W = self.search_layer(q, ep, 1, level)
                ep = W[0]

            for level in range(min(L, l), -1, -1):
                W = self.search_layer(q, ep, self.ef_construction, level)
                neighbors, dists = self.select_neighbors(idx, W, self.M, level)

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
                        neighbor_new_connections, neighbor_new_dist = self.select_neighbors(neighbor, neighbor_connections, maxM, level)

                        # set new neighborhood of neighbor
                        self.graph[level][neighbor].clear()
                        for n, d in zip(neighbor_new_connections, neighbor_new_dist):
                            self.graph[level][neighbor][n] = d
                            if n == neighbor:
                                print("BUG!")

                ep = W[0]

        if L < l:
            self.entry_point = idx
            self.max_level = l
            for i in range(L, l):
                self.graph.append({idx: {}})

    def search_layer(self,
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
                    f_dist, f_id = W[0]

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
            W = self.search_layer(q, ep, 1, level)
            ep = W[0]

        W = self.search_layer(q, ep, max(K, self.ef), 0)
        return W[:K]

    def select_neighbors(self,
                        #  q: ArrayLike,
                         q_id: int,
                         W: list[int],
                         M: int,
                         level: int) -> tuple[list[int], list[float]]:
        proba = self.rng.uniform(0.0, 1.0)
        if proba <= 0.5:
            return self.select_neighbors_simple(q_id, W, M)
        return self.select_neighbors_heuristic(q_id, W, M, level)

    def select_neighbors_simple(self,
                                # q: ArrayLike,
                                q_id: int,
                                C: list[int],
                                M: int) -> tuple[list[int], list[float]]:
        nearest_neighbors = []

        for c in C:
            dist = -self.distance(self.data[c], self.data[q_id])
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
                                #    q: ArrayLike,
                                   q_id: int,
                                   C: list[int],
                                   M: int,
                                   level: int,
                                   extend_candidates: bool = True,
                                   keep_pruned_connections: bool = True) -> tuple[list[int], list[float]]:
        R = []
        W = set(C)
        W_queue = []
        for c in C:
            heapq.heappush(W_queue, (self.distance(self.data[c], self.data[q_id]), c))

        if extend_candidates:
            temp_W = W.copy()
            for e in temp_W:
                for en_id, en_dist in self.graph[level][e].items():
                    if en_id not in W and en_id != q_id:
                        W.add(en_id)
                        heapq.heappush(W_queue, (self.distance(self.data[en_id], self.data[q_id]), en_id))

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

    def visualize_layer(self, l: int):
        """
        Helper function of visualize_layers
        """
        if l < 0 or l > self.max_level:
            print('Layer\'s level is out of range!')
            return

        fig = plt.figure(figsize=(10, 10))
        if fig and fig.canvas and fig.canvas.manager:
            fig.canvas.manager.set_window_title(f'HNSW Graph - Layer {l}')

        G = nx.Graph(name=f'Layer {l}')

        for node_id, node_neighbors in self.graph[l].items():
            G.add_node(node_id)
            for neighbor_id in node_neighbors.keys():
                G.add_edge(node_id, neighbor_id)

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos)

        plt.title(f'HNSW Graph - Layer {l}')
        fig.canvas.draw_idle()

    def visualize_layers(self, layers: list[int] | None = None):
        if layers == None:
            layers = [i for i in range(self.max_level + 1)]

        plt.ion()

        for l in layers:
            self.visualize_layer(l)

        plt.ioff()
        plt.show(block=True)

    def visualize_hierarchical_graph(self):
        G0 = nx.Graph()
        for node_id in self.graph[0].keys():
            G0.add_node(node_id)
            for neighbor_id in self.graph[0][node_id].keys():
                G0.add_edge(node_id, neighbor_id)
        pos_2d = nx.spring_layout(G0, dim=2, seed=42)

        all_node_ids = set(self.graph[0].keys())
        
        # List of Traces
        data_traces = []
        
        # Vertical trace
        vertical_edge_x, vertical_edge_y, vertical_edge_z = [], [], []

        # Build trace per layer l
        for l in range(self.max_level + 1):
            layer_node_data = []
            layer_edge_x, layer_edge_y, layer_edge_z = [], [], []

            for node_id in all_node_ids:
                max_node_level = self.level[node_id]
                x, y = pos_2d.get(node_id, (0, 0))

                if l <= max_node_level:
                    layer_node_data.append({
                        'id': node_id, 
                        'x': x, 
                        'y': y, 
                        'level': l
                    })

                # === B. DỮ LIỆU CẠNH NGANG (Tầng l) ===
                if node_id in self.graph[l]:
                    for neighbor_id in self.graph[l][node_id].keys():
                        if node_id >= neighbor_id: 
                            continue

                        if neighbor_id in pos_2d:
                            x_n, y_n = pos_2d[neighbor_id]
                            z = l
                            
                            layer_edge_x.extend([x, x_n, None])
                            layer_edge_y.extend([y, y_n, None])
                            layer_edge_z.extend([z, z, None])

                # Vertical edge trace
                if l < max_node_level:
                    vertical_edge_x.extend([x, x, None])
                    vertical_edge_y.extend([y, y, None])
                    vertical_edge_z.extend([l + 1, l, None])

            # Create trace for current layer l
            # Node trace at layer l
            if layer_node_data:
                df_layer_nodes = pd.DataFrame(layer_node_data)
                data_traces.append(go.Scatter3d(
                    x=df_layer_nodes['x'], y=df_layer_nodes['y'], z=df_layer_nodes['level'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=5,
                        color='cyan',
                        line=dict(color='black', width=0.5)),
                    name=f'Nodes (Layer {l})',
                    legendgroup=f'Layer_{l}',
                    hoverinfo='text',
                    # Description of node
                    text=[f"ID: {id}<br>Layer: {l}" for id, l in zip(df_layer_nodes['id'], df_layer_nodes['level'])]
                ))

            # Edge trace at layer l
            if layer_edge_x:
                data_traces.append(go.Scatter3d(
                    x=layer_edge_x, y=layer_edge_y, z=layer_edge_z,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines',
                    name=f'Edges (Layer {l})',
                    legendgroup=f'Layer_{l}'
                ))

        # Create vertical trace
        if vertical_edge_x:
            data_traces.append(go.Scatter3d(
                x=vertical_edge_x, y=vertical_edge_y, z=vertical_edge_z,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Vertical Connections',
                legendgroup='Vertical'
            ))

        if self.entry_point != -1:
            x, y = pos_2d.get(self.entry_point, (0, 0))
            df_entry = pd.DataFrame([{
                'id': self.entry_point,
                'x': x,
                'y': y,
                'level': self.max_level
            }])
            data_traces.append(go.Scatter3d(
                x=df_entry['x'], y=df_entry['y'], z=df_entry['level'],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=5,
                    color='blue', 
                    line=dict(color='black', width=0.5)),
                name=f'Entry Point',
                legendgroup=f'Layer_{self.max_level}',
                hoverinfo='text',
                # Description of entry point
                text=[f"Entry Point's ID: {int(df_entry.iloc[0]['id'])}<br>Layer: {int(df_entry.iloc[0]['level'])}"]
            ))

        # Show figure
        fig = go.Figure(data=data_traces,
                        layout=go.Layout(
                            title=f'HNSW 3D Graph Visualization (Max Level: {self.max_level})',
                            scene=dict(
                                xaxis=dict(title='X (Layout Position)', showbackground=False, showgrid=False, zeroline=False, visible=False),
                                yaxis=dict(title='Y (Layout Position)', showbackground=False, showgrid=False, zeroline=False, visible=False),
                                zaxis=dict(title='Z (Layer Index)', showbackground=False, showgrid=False, zeroline=False, nticks=self.max_level + 2) # Hiển thị ticks cho từng tầng
                            ),
                            hovermode='closest'
                        ))

        fig.show()

if __name__ == '__main__':
    train_data = np.random.rand(100, 100)
    query_data = np.random.rand(3, 100)

    max_elements = 50
    dim = 100
    M = 3
    ef_construction = 6

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

    # print(index.max_level)

    W = index.knn_search(query_data[0], 10)
    dists = [l2_distance(query_data[0], train_data[w])**2 for w in W]

    # print(np.array(W))
    # print(np.array(dists))

    index.visualize_hierarchical_graph()

    # print('='*10)
