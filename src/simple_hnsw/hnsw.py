import numpy as np
import pandas as pd
import heapq

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, no_update

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
        """
        Initialize the HNSW index.

        Args:
            max_elements (int): Maximum number of elements in the index.
            M (int, optional): The number of bi-directional links created for every new element during construction. Defaults to 16.
            ef_construction (int, optional): The size of the dynamic list for the nearest neighbors (used during the search). Defaults to 200.
            random_seed (int, optional): Seed for random number generator. Defaults to 12345.
        """
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
        """
        Insert multiple items into the index.

        Args:
            data (ArrayLike): The data to insert. Should be a 2D array-like structure.
            visualize (bool, optional): Whether to visualize the insertion process (not implemented in this method, passed to insert). Defaults to False.
        """
        data = np.atleast_2d(data)
        for d in data:
            self.insert(d, visualize)

    def insert(self, q: ArrayLike, visualize: bool = False) -> None:
        """
        Insert a single item into the index.

        Args:
            q (ArrayLike): The query vector to insert.
            visualize (bool, optional): Whether to visualize the insertion process. Defaults to False.
        """
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
                W = self.seach_layer(q, ep, 1, level)
                ep = W[0]

            for level in range(min(L, l), -1, -1):
                W = self.seach_layer(q, ep, self.ef_construction, level)
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

    def seach_layer(self,
                    q: ArrayLike,
                    ep: int,
                    ef: int,
                    level: int,
                    logger=None) -> list[int]:
        """
        Search for the nearest neighbors in a specific layer.

        Args:
            q (ArrayLike): The query vector.
            ep (int): The entry point for the search in this layer.
            ef (int): The size of the dynamic list for the nearest neighbors.
            level (int): The level of the layer to search in.
            logger (callable, optional): A logging function for visualization events. Defaults to None.

        Returns:
            list[int]: A list of nearest neighbor indices found in this layer.
        """
        visited = set([ep])

        # These are containers of heap
        # each element is a tuple of (dist, id)
        candidates = [(self.distance(self.data[ep], q), ep)]
        W = [(-candidates[0][0], ep)] # negate first element to perform as max heap

        if logger:
            logger({
                'event': 'init_search_layer',
                'layer': level,
                'ep': ep
            })

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            f_dist, f_id = W[0]

            if c_dist > -f_dist:
                break

            if logger:
                logger({
                    'event': 'visit_node',
                    'current_node': c_id
                })

            for e_id, e_dist in self.graph[level][c_id].items():
                if e_id not in visited:
                    visited.add(e_id)
                    e_dist = self.distance(self.data[e_id], q)
                    f_dist, f_id = W[0]

                    if logger:
                        logger({
                            'event': 'consider_neighbor',
                            'current_node': c_id,
                            'neighbor': e_id
                        })

                    if e_dist < -f_dist or len(W) < ef:
                        heapq.heappush(candidates, (e_dist, e_id))
                        heapq.heappush(W, (-e_dist, e_id))

                        if logger:
                            logger({
                                'event': 'accept_neighbor',
                                'current_node': c_id,
                                'neighbor': e_id
                            })

                        if len(W) > ef:
                            reject_dist, reject_id = heapq.heappop(W)
                            if logger:
                                logger({
                                    'event': 'reject_node',
                                    'current_node': c_id,
                                    'reject_node': reject_id
                                })
                    elif logger:
                        logger({
                            'event': 'reject_neighbor',
                            'current_node': c_id,
                            'neighbor': e_id
                        })

        nearest_neighbors = heapq.nlargest(ef, W)
        nearest_neighbors = [nn[1] for nn in nearest_neighbors]

        return nearest_neighbors

    def knn_search(self, q: ArrayLike, K: int = 1, logger=None) -> list[int]:
        """
        Perform K-Nearest Neighbors search.

        Args:
            q (ArrayLike): The query vector.
            K (int, optional): The number of nearest neighbors to return. Defaults to 1.
            logger (callable, optional): A logging function for visualization events. Defaults to None.

        Returns:
            list[int]: A list of indices of the K nearest neighbors.
        """
        W = []
        ep = self.entry_point
        L = self.max_level

        if logger:
            logger({
                'event': 'init_knn_search',
                'entry_point': ep,
                'max_level': L
            })

        for level in range(L, 0, -1):
            if logger and level != L:
                logger({
                    'event': 'layer_transition',
                    'from_layer': level + 1,
                    'to_layer': level,
                    'ep': ep
                })
            W = self.seach_layer(q, ep, 1, level, logger=logger)
            ep = W[0]

        if logger:
            logger({
                'event': 'layer_transition',
                'from_layer': 1,
                'to_layer': 0,
                'ep': ep
            })

        W = self.seach_layer(q, ep, max(K, self.ef), 0, logger=logger)

        if logger:
            reject_nodes = []
            for i in range(K, len(W)):
                reject_nodes.append(W[i])

            logger({
                'event': 'reject_nodes',
                'reject_nodes': reject_nodes
            })

        return W[:K]

    def select_neighbors(self,
                        #  q: ArrayLike,
                         q_id: int,
                         W: list[int],
                         M: int,
                         level: int) -> tuple[list[int], list[float]]:
        """
        Select neighbors for a node.

        Args:
            q_id (int): The ID of the query node.
            W (list[int]): A list of candidate neighbor IDs.
            M (int): The maximum number of neighbors to select.
            level (int): The level at which to select neighbors.

        Returns:
            tuple[list[int], list[float]]: A tuple containing the selected neighbor IDs and their distances.
        """
        proba = self.rng.uniform(0.0, 1.0)
        if proba <= 0.5:
            return self.select_neighbors_simple(q_id, W, M)
        return self.select_neighbors_heuristic(q_id, W, M, level)

    def select_neighbors_simple(self,
                                # q: ArrayLike,
                                q_id: int,
                                C: list[int],
                                M: int) -> tuple[list[int], list[float]]:
        """
        Select neighbors using a simple heuristic (closest ones).

        Args:
            q_id (int): The ID of the query node.
            C (list[int]): A list of candidate neighbor IDs.
            M (int): The maximum number of neighbors to select.

        Returns:
            tuple[list[int], list[float]]: A tuple containing the selected neighbor IDs and their distances.
        """
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
        """
        Select neighbors using the HNSW heuristic.

        Args:
            q_id (int): The ID of the query node.
            C (list[int]): A list of candidate neighbor IDs.
            M (int): The maximum number of neighbors to select.
            level (int): The level at which to select neighbors.
            extend_candidates (bool, optional): Whether to extend candidates by checking their neighbors. Defaults to True.
            keep_pruned_connections (bool, optional): Whether to keep pruned connections. Defaults to True.

        Returns:
            tuple[list[int], list[float]]: A tuple containing the selected neighbor IDs and their distances.
        """
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

    # ===================================================
    # =============== VISUALIZE METHODS =================
    # ===================================================

    def visualize_layer(self, l: int):
        """
        Visualize a specific layer of the HNSW graph using NetworkX and Matplotlib.

        Args:
            l (int): The level of the layer to visualize.
        """
        if l < 0 or l > self.max_level:
            print("Layer's level is out of range!")
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
        """
        Visualize multiple layers of the HNSW graph.

        Args:
            layers (list[int] | None, optional): A list of layer levels to visualize. If None, all layers are visualized. Defaults to None.
        """
        if layers == None:
            layers = [i for i in range(self.max_level + 1)]

        plt.ion()

        for l in layers:
            self.visualize_layer(l)

        plt.ioff()
        plt.show(block=True)

    def _compute_layout(self):
        """
        Compute 2D positions for nodes based on layer 0 connections.
        Returns a dictionary mapping (node_id, layer) to (x, y, z).
        """
        G0 = nx.Graph()
        if self.graph:
            for node_id in self.graph[0].keys():
                G0.add_node(node_id)
                for neighbor_id in self.graph[0][node_id].keys():
                    G0.add_edge(node_id, neighbor_id)

        pos_2d = nx.spring_layout(G0, dim=2, seed=42)

        node_positions = {}
        if self.graph:
            all_node_ids = set(self.graph[0].keys())
            for node_id in all_node_ids:
                x, y = pos_2d.get(node_id, (0, 0))
                max_node_level = self.level[node_id]
                for l in range(max_node_level + 1):
                    node_positions[(node_id, l)] = (x, y, l)
        
        return node_positions

    def _create_base_traces(self, node_positions):
        """
        Create static base traces for the HNSW graph visualization.
        """
        x_nodes, y_nodes, z_nodes = [], [], []
        x_edges, y_edges, z_edges = [], [], []
        x_vedges, y_vedges, z_vedges = [], [], []

        text_nodes = []

        all_node_ids = set()
        if self.graph:
            all_node_ids = set(self.graph[0].keys())

        for l in range(self.max_level + 1):
            for node_id in all_node_ids:
                if (node_id, l) in node_positions:
                    x, y, z = node_positions[(node_id, l)]
                    x_nodes.append(x)
                    y_nodes.append(y)
                    z_nodes.append(z)
                    text_nodes.append(f"ID: {node_id}<br>Layer: {l}")

                    # Horizontal edges
                    if node_id in self.graph[l]:
                        for neighbor_id in self.graph[l][node_id]:
                            if node_id < neighbor_id:
                                if (neighbor_id, l) in node_positions:
                                    xn, yn, zn = node_positions[(neighbor_id, l)]
                                    x_edges.extend([x, xn, None])
                                    y_edges.extend([y, yn, None])
                                    z_edges.extend([z, zn, None])

                    # Vertical edges
                    if l < self.max_level and (node_id, l+1) in node_positions:
                        xn, yn, zn = node_positions[(node_id, l+1)]
                        x_vedges.extend([x, x, None])
                        y_vedges.extend([y, y, None])
                        z_vedges.extend([l, l+1, None])

        traces = [
            # 1. Base Edges
            go.Scatter3d(x=x_edges, y=y_edges, z=z_edges, mode='lines', 
                         line=dict(color='#888', width=1), hoverinfo='none', name='Edges'),
            # 2. Base Vertical Edges
            go.Scatter3d(x=x_vedges, y=y_vedges, z=z_vedges, mode='lines', 
                         line=dict(color='#888', width=1, dash='dash'), hoverinfo='none', name='Vertical Edges '),
            # 3. Base Nodes
            go.Scatter3d(x=x_nodes, y=y_nodes, z=z_nodes, mode='markers', 
                         marker=dict(size=4, color='cyan', line=dict(color='black', width=0.5)), 
                         text=text_nodes, hoverinfo='text', name='Nodes')
        ]

        # Entry Point
        if self.entry_point != -1:
            if (self.entry_point, self.max_level) in node_positions:
                x, y, z = node_positions[(self.entry_point, self.max_level)]
                traces.append(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers',
                    marker=dict(symbol='circle', size=6, color='blue', line=dict(color='black', width=0.5)),
                    name='Entry Point',
                    text=[f"Entry Point ID: {self.entry_point}"],
                    hoverinfo='text'
                ))

        return traces

    def visualize_hierarchical_graph(self):
        """
        Visualize the entire hierarchical graph in 3D using Dash.
        """
        # Create layout and traces
        node_positions = self._compute_layout()
        base_traces = self._create_base_traces(node_positions)

        # Modify base traces to disable hover on nodes
        for trace in base_traces:
            if trace.name in ['Nodes', 'Entry Point']:
                trace.hoverinfo = 'skip'

        # Create Invisible Hitboxes
        hitbox_x, hitbox_y, hitbox_z = [], [], []
        hitbox_customdata = []

        for (node_id, level), (x, y, z) in node_positions.items():
            hitbox_x.append(x)
            hitbox_y.append(y)
            hitbox_z.append(z)
            hitbox_customdata.append(node_id)

        hitbox_trace = go.Scatter3d(
            x=hitbox_x, y=hitbox_y, z=hitbox_z,
            mode='markers',
            marker=dict(
                symbol='circle',
                size=30,  # Hitbox size
                color='white',
                opacity=0.0,   # Invisible
            ),
            customdata=hitbox_customdata,
            hovertemplate="<b>ID: %{customdata}</b><br>Level: %{z}<extra></extra>",
            name='Hitbox'
        )

        fig_traces = base_traces + [hitbox_trace]

        # Create Figure
        layout = go.Layout(
            title=f'HNSW Interactive Dashboard',
            scene=dict(
                xaxis=dict(title='', showbackground=False, showgrid=False, zeroline=False, visible=False),
                yaxis=dict(title='', showbackground=False, showgrid=False, zeroline=False, visible=False),
                zaxis=dict(title='Level', showbackground=False, showgrid=False, zeroline=False, nticks=self.max_level + 2),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode='closest',
            uirevision='constant'
        )

        base_fig = go.Figure(data=fig_traces, layout=layout)

        # Initialize Dash
        app = Dash(__name__)

        app.layout = html.Div([
            html.Title("HNSW Interactive Dashboard"),
            dcc.Graph(
                id='hnsw-viz',
                figure=base_fig,
                style={'height': '95vh'},
                clear_on_unhover=True
            )
        ])

        @app.callback(
            Output('hnsw-viz', 'figure'),
            Input('hnsw-viz', 'hoverData')
        )
        def update_highlight(hoverData):
            if hoverData is None:
                return base_fig

            try:
                point = hoverData['points'][0]
                if 'customdata' not in point: return no_update
                node_id = point['customdata']
            except:
                return no_update

            # Create Highlight
            new_fig = go.Figure(base_fig)

            if (node_id, 0) not in node_positions:
                return no_update

            x, y, _ = node_positions[(node_id, 0)]
            max_l = self.level[node_id]

            hl_x = [x] * (max_l + 1)
            hl_y = [y] * (max_l + 1)
            hl_z = list(range(max_l + 1))

            # Highlight Vertical Line
            new_fig.add_trace(go.Scatter3d(
                x=hl_x, y=hl_y, z=hl_z,
                mode='lines',
                line=dict(color='yellow', width=8),
                opacity=0.8, hoverinfo='skip', name='Highlight Line'
            ))

            # Highlight Node Halo
            new_fig.add_trace(go.Scatter3d(
                x=hl_x, y=hl_y, z=hl_z,
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=12,
                    color='yellow',
                    opacity=0.6,
                    line=dict(width=0)
                ),
                hoverinfo='skip', name='Highlight Node'
            ))

            return new_fig

        app.run(debug=True, use_reloader=False)

    def visualize_search(self, q: ArrayLike, k: int = 1) -> list[int]:
        """
        Visualize the search process for a query vector in 3D.

        Args:
            q (ArrayLike): The query vector.
            k (int, optional): The number of nearest neighbors to search for. Defaults to 1.

        Returns:
            list[int]: The indices of the nearest neighbors found.
        """
        q = np.array(q)
        search_log = []

        def logger(event):
            search_log.append(event)

        # 1. Perform Search with Logging
        # We use the instrumented knn_search which calls search_layer
        result_ids = self.knn_search(q, k, logger=logger)

        # 2. Visualization with Plotly
        node_positions = self._compute_layout()
        base_traces = self._create_base_traces(node_positions)

        # Helper to get coords
        def get_coords(items):
            xs, ys, zs, items_text = [], [], [], []
            for node_id, l in items:
                x, y, z = node_positions[(node_id, l)]
                xs.append(x)
                ys.append(y)
                zs.append(z)
                items_text.append(f"ID: {node_id}<br>Layer: {l}")
            return xs, ys, zs, items_text

        def get_edge_coords(items):
            xs, ys, zs = [], [], []
            for (u, v), l in items:
                ux, uy, uz = node_positions[(u, l)]
                vx, vy, vz = node_positions[(v, l)]
                xs.extend([ux, vx, None])
                ys.extend([uy, vy, None])
                zs.extend([uz, vz, None])
            return xs, ys, zs

        def get_vedge_coords(items):
            xs, ys, zs = [], [], []
            for (u1, l1), (u2, l2) in items:
                x1, y1, z1 = node_positions[(u1, l1)]
                x2, y2, z2 = node_positions[(u2, l2)]
                xs.extend([x1, x2, None])
                ys.extend([y1, y2, None])
                zs.extend([z1, z2, None])
            return xs, ys, zs

        # Define Fixed Traces (14 Traces) - MUST MATCH create_traces order
        def create_traces(
            sel_e_x, sel_e_y, sel_e_z,
            sel_ve_x, sel_ve_y, sel_ve_z,
            rej_e_x, rej_e_y, rej_e_z,
            con_e_x, con_e_y, con_e_z,
            sel_n_x, sel_n_y, sel_n_z, sel_n_text,
            rej_n_x, rej_n_y, rej_n_z, rej_n_text,
            con_n_x, con_n_y, con_n_z, con_n_text,
            cur_x, cur_y, cur_z, cur_text
        ):
            # Start with base traces (0, 1, 2)
            traces = list(base_traces) # Copy base traces

            # Add dynamic traces
            traces.extend([
                go.Scatter3d(x=sel_e_x, y=sel_e_y, z=sel_e_z, mode='lines', line=dict(color='green', width=4), name='Select Edge'),
                go.Scatter3d(x=sel_ve_x, y=sel_ve_y, z=sel_ve_z, mode='lines', line=dict(color='green', width=4, dash='dash'), name='Select Edge'),
                go.Scatter3d(x=rej_e_x, y=rej_e_y, z=rej_e_z, mode='lines', line=dict(color='red', width=3), name='Reject Edge'),
                go.Scatter3d(x=con_e_x, y=con_e_y, z=con_e_z, mode='lines', line=dict(color='yellow', width=3), name='Consider Edge'),

                go.Scatter3d(x=sel_n_x, y=sel_n_y, z=sel_n_z, mode='markers', marker=dict(size=6, color='green', symbol='circle', line=dict(color='black', width=0.5)), text=sel_n_text, hoverinfo='text', name='Select'),
                go.Scatter3d(x=rej_n_x, y=rej_n_y, z=rej_n_z, mode='markers', marker=dict(size=6, color='red', symbol='circle', line=dict(color='black', width=0.5)), text=rej_n_text, hoverinfo='text', name='Reject'),
                go.Scatter3d(x=con_n_x, y=con_n_y, z=con_n_z, mode='markers', marker=dict(size=6, color='yellow', symbol='circle', line=dict(color='black', width=0.5)), text=con_n_text, hoverinfo='text', name='Consider'),
                go.Scatter3d(x=cur_x, y=cur_y, z=cur_z, mode='markers', marker=dict(size=10, color='green', symbol='circle', line=dict(color='black', width=1)), text=cur_text, hoverinfo='text', name='Current')
            ])
            return traces

        # Animation Frames
        frames = []

        current_node = -1
        current_layer = -1
        current_W_set = set()

        for i, log in enumerate(search_log):
            event = log['event']

            # Current state
            current_selected_edge = set()
            current_selected_vedge = set()
            current_considered_edge = set()
            current_rejected_edge = set()

            current_considered_node = set()
            current_rejected_node = set()

            if event == 'init_knn_search':
                current_node = log['entry_point']
                current_layer = log['max_level']
                current_W_set = {(current_node, current_layer)}

            elif event == 'layer_transition':
                u = log['ep']
                l_from = log['from_layer']
                l_to = log['to_layer']

                current_selected_vedge.add(((u, l_from), (u, l_to)))
                current_layer = l_to
                current_node = u
                current_W_set = {(current_node, current_layer)}

            elif event == 'init_search_layer':
                current_layer = log['layer']
                ep = log['ep']
                current_node = ep
                current_W_set = {(current_node, current_layer)}

            elif event == 'visit_node':
                current_node = log['current_node']

            elif event == 'consider_neighbor':
                neighbor = log['neighbor']
                u, v = sorted((current_node, neighbor))

                current_considered_edge.add(((u, v), current_layer))
                current_considered_node.add((neighbor, current_layer))

            elif event == 'accept_neighbor':
                neighbor = log['neighbor']
                u, v = sorted((current_node, neighbor))

                current_selected_edge.add(((u, v), current_layer))
                current_W_set.add((neighbor, current_layer))

            elif event == 'reject_neighbor':
                neighbor = log['neighbor']
                u, v = sorted((current_node, neighbor))

                current_rejected_node.add((neighbor, current_layer))
                current_rejected_edge.add(((u, v), current_layer))

            elif event == 'reject_node':
                reject_node = log['reject_node']
                u, v = sorted((current_node, reject_node))

                current_rejected_node.add((reject_node, current_layer))
                current_W_set.remove((reject_node, current_layer))

            elif event == 'reject_nodes':
                reject_nodes = log['reject_nodes']
                for id in reject_nodes:
                    current_rejected_node.add((id, current_layer))

            # Calculate Coords for all dynamic traces
            sel_e_x, sel_e_y, sel_e_z = get_edge_coords(current_selected_edge)
            sel_ve_x, sel_ve_y, sel_ve_z = get_vedge_coords(current_selected_vedge)
            rej_e_x, rej_e_y, rej_e_z = get_edge_coords(current_rejected_edge)
            con_e_x, con_e_y, con_e_z = get_edge_coords(current_considered_edge)

            sel_n_x, sel_n_y, sel_n_z, sel_n_text = get_coords(current_W_set)
            rej_n_x, rej_n_y, rej_n_z, rej_n_text = get_coords(current_rejected_node)
            con_n_x, con_n_y, con_n_z, con_n_text = get_coords(current_considered_node)
            cur_x, cur_y, cur_z, cur_text = get_coords([(current_node, current_layer)])

            frame_traces = create_traces(
                sel_e_x, sel_e_y, sel_e_z,
                sel_ve_x, sel_ve_y, sel_ve_z,
                rej_e_x, rej_e_y, rej_e_z,
                con_e_x, con_e_y, con_e_z,
                sel_n_x, sel_n_y, sel_n_z, sel_n_text,
                rej_n_x, rej_n_y, rej_n_z, rej_n_text,
                con_n_x, con_n_y, con_n_z, con_n_text,
                cur_x, cur_y, cur_z, cur_text
            )

            frames.append(go.Frame(data=frame_traces, name=f'step_{i}'))

        # Final Result Frame
        final_x, final_y, final_z, final_text = [], [], [], []
        for r_id in result_ids:
            rx, ry, rz = node_positions[(r_id, 0)]
            final_x.append(rx)
            final_y.append(ry)
            final_z.append(rz)
            final_text.append(f"ID: {r_id}<br>Layer: 0")

        # Final frame: Highlight result nodes
        final_traces = create_traces(
            [], [], [], # Sel E
            [], [], [], # Sel VE
            [], [], [], # Rej E
            [], [], [], # Con E
            [], [], [], [], # Sel N
            [], [], [], [], # Rej N
            [], [], [], [], # Con N
            final_x, final_y, final_z, final_text # Cur (reused for result)
        )
        # Override name/color for final result if needed, but reusing 'Current Focus' (Green) is okay for Top K

        frames.append(go.Frame(data=final_traces, name='final'))

        # Initial Layout
        initial_traces = create_traces(
            [], [], [], # Sel E
            [], [], [], # Sel VE
            [], [], [], # Rej E
            [], [], [], # Con E
            [], [], [], [], # Sel N
            [], [], [], [], # Rej N
            [], [], [], [], # Con N
            [], [], [], []  # Cur N
        )

        fig = go.Figure(
            data=initial_traces,
            layout=go.Layout(
                title=f'HNSW Search Visualization (k={k})',
                scene=dict(
                    xaxis=dict(title='X', showbackground=False, showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(title='Y', showbackground=False, showgrid=False, zeroline=False, visible=False),
                    zaxis=dict(title='Layer', showbackground=False, showgrid=False, zeroline=False, nticks=self.max_level + 2)
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                }]
            ),
            frames=frames
        )

        fig.show()

        return result_ids

if __name__ == '__main__':
    train_data = np.random.rand(10, 100)
    query_data = np.random.rand(3, 100)

    max_elements = 50
    dim = 100
    M = 3
    ef_construction = 6

    index = HNSW('l2', dim)
    index.init_index(max_elements, M, ef_construction)

    index.insert_items(train_data)

    index.ef = 5

    l = index.max_level
    ep = index.entry_point

    W = index.knn_search(query_data[0], 10)
    dists = [l2_distance(query_data[0], train_data[w])**2 for w in W]

    index.visualize_hierarchical_graph()
    index.visualize_search(query_data[0], k=3)
