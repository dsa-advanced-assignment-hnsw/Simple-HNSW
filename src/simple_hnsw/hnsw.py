import numpy as np
import pandas as pd
import heapq

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, no_update

from typing import Literal, Any
from numpy.typing import ArrayLike, NDArray

from .distance_metrics import l2_distance, cosine_distance

class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) Graph Implementation.

    This class implements the HNSW algorithm for Approximate Nearest Neighbor (ANN) search.
    It constructs a multi-layer graph structure where the bottom layer (Layer 0) contains 
    all data points, and upper layers serve as sparse "expressways" for logarithmic 
    scaling of the search process.

    The algorithm approximates a randomized skip-list structure generalized for 
    high-dimensional nearest neighbor search.

    Attributes:
        dim (int): Dimensionality of the data vectors.
        distance (callable): The distance metric function (L2 or Cosine).
        max_elements (int): The hard limit on the capacity of the index.
        M (int): The target number of bi-directional links for each node at layers > 0.
            This parameter controls the trade-off between graph density/recall and memory/speed.
        maxM (int): Maximum number of connections allowed per node for layers > 0.
        maxM0 (int): Maximum number of connections allowed per node for layer 0.
            Layer 0 requires higher connectivity to ensure global navigability.
        m_L (float): Level generation normalization factor (1 / ln(M)).
            Used to determine the maximum layer of a new node based on an exponential distribution.
        ef_construction (int): Size of the dynamic candidate list (beam width) during index construction.
            Higher values improve graph quality (recall) at the cost of slower indexing.
        ef (int): Size of the dynamic candidate list during query time.
            Can be adjusted dynamically to trade off search speed for recall.
        graph (list[dict[int, dict]]): Adjacency list representation of the multi-layer graph.
            Structure: graph[level][node_id] -> {neighbor_id: distance}
        data (list[NDArray]): Internal storage for the vector data (row-major).
        level (list[int]): Stores the maximum layer assigned to each node index.
        cur_element_count (int): Current number of elements stored in the index.
        entry_point (int): The global entry point node ID (at the highest layer).
        max_level (int): The current maximum layer depth of the graph.
        rng (np.random.Generator): Random number generator for reproducible level assignment.
    """
    
    def __init__(self, space: Literal['l2', 'cosine'], dim: int) -> None:
        """
        Initializes the HNSW instance with dimensionality and metric space.

        Args:
            space (Literal['l2', 'cosine']): The metric space to use. 
                'l2' for Euclidean distance, 'cosine' for Cosine distance.
            dim (int): The dimensionality of the vectors.
        """
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
        Configures the index hyperparameters and initializes storage.

        Args:
            max_elements (int): The maximum number of elements this index can hold.
            M (int, optional): The number of neighbors to connect to in the graph. Defaults to 16.
            ef_construction (int, optional): The size of the candidate queue during insertion.
                Defaults to 200.
            random_seed (int, optional): Seed for random level generation. Defaults to 12345.
        """
        self.max_elements = max_elements
        self.M = M
        self.maxM = M
        # Layer 0 usually connects to more neighbors (2*M) to adhere to the Small World theory
        # ensuring the graph remains connected.
        self.maxM0 = M * 2
        
        # Normalization factor for level generation: -ln(uniform) * m_L
        # Ensures that the number of nodes decays exponentially with the layer level.
        self.m_L = 1 / np.log(M)
        
        self.ef_construction = max(ef_construction, M)
        self.ef = 10 # Default query-time parameter

        self.graph: list[dict[int, dict]] = [] # graph[level][i] is a dict of (neighbor_id, dist)
        self.data: list[NDArray] = [] # Storage for embedded vectors
        self.level: list[int] = [] # Max level for each node

        self.cur_element_count = 0
        self.entry_point = -1
        self.max_level = -1

        self.rng = np.random.default_rng(random_seed)

    def insert_items(self, data: ArrayLike, visualize: bool = False) -> None:
        """
        Batch inserts multiple vectors into the index.

        Args:
            data (ArrayLike): Input data, typically an (N, dim) array.
            visualize (bool, optional): Trigger for visualization hooks. Defaults to False.
        """
        data = np.atleast_2d(data)
        for d in data:
            self.insert(d, visualize)

    def insert(self, q: ArrayLike, visualize: bool = False) -> None:
        """
        Inserts a single vector into the HNSW graph.

        The insertion procedure follows two phases:
        1. **Zoom-in (Coarse Search):** Starts from the top layer down to the node's assigned layer `l`.
           Uses greedy search to find the nearest entry point at layer `l`.
        2. **Construction (Fine Search):** From layer `l` down to 0, finds `ef_construction` 
           nearest neighbors, selects `M` diverse neighbors using heuristics, and adds 
           bidirectional connections.

        Args:
            q (ArrayLike): The query vector to insert.
            visualize (bool, optional): Reserved for visualization callbacks. Defaults to False.
        """
        q = np.array(q)

        W = []
        ep = self.entry_point
        L = self.max_level
        
        # Sample the level for the new element using an exponential distribution
        l = int(-np.log(self.rng.uniform(0.0, 1.0)) * self.m_L)

        self.data.append(q)
        self.level.append(l)
        idx = self.cur_element_count
        self.cur_element_count += 1

        if self.entry_point != -1:
            # Phase 1: Greedily traverse from top layer L down to l+1
            # to find the closest entry point for the insertion layer.
            for level in range(L, l, -1):
                W = self.search_layer(q, ep, 1, level)
                ep = W[0]

            # Phase 2: Insert the element at layer l down to 0
            for level in range(min(L, l), -1, -1):
                W = self.search_layer(q, ep, self.ef_construction, level)
                neighbors, dists = self.select_neighbors(idx, W, self.M, level)

                # Establish bidirectional connections
                for neighbor, dist in zip(neighbors, dists):
                    self._add_bidirectional_connection(idx, neighbor, dist, level)

                # Enforce the maxM constraint on connections for updated neighbors
                for neighbor in neighbors:
                   self._prune_connections(neighbor, level)

                # The nearest neighbor found becomes the entry point for the next layer down
                ep = W[0]

        # Update the global entry point if the new node is at a higher level than current max
        if L < l:
            self.entry_point = idx
            self.max_level = l
            for i in range(L, l):
                self.graph.append({idx: {}})

    def _add_bidirectional_connection(self, u: int, v: int, dist: float, level: int):
        """Helper to add an undirected edge (u, v) with weight `dist` at a specific level."""
        if u not in self.graph[level]:
            self.graph[level][u] = {}
        self.graph[level][u][v] = dist
        self.graph[level][v][u] = dist

    def _prune_connections(self, u: int, level: int):
        """
        Shrinks the connection list of node `u` if it exceeds `maxM`.
        
        Uses the heuristic selection algorithm to keep the most diverse/closest neighbors
        and removes the rest.
        """
        connections = list(self.graph[level][u].keys())
        maxM = self.maxM if level > 0 else self.maxM0

        if len(connections) > maxM:
            new_connections, new_dists = self.select_neighbors(u, connections, maxM, level)

            kept_neighbors = set(new_connections)
            for n in connections:
                if n in kept_neighbors:
                    # Update distance (redundant if unchanged, but safe)
                    self._add_bidirectional_connection(u, n, self.graph[level][u][n], level)
                else:
                    # Remove the edge from both u and the neighbor n
                    self.graph[level][n].pop(u)
                    self.graph[level][u].pop(n)

    def search_layer(self,
                    q: ArrayLike,
                    ep: int,
                    ef: int,
                    level: int,
                    logger=None) -> list[int]:
        """
        Performs a greedy Best-First Search on a specific layer.

        Maintains two structures:
        1. `candidates`: A min-heap of nodes to explore (frontier).
        2. `W`: A max-heap (simulated with negative distances) of the top-`ef` found neighbors.

        The search stops when the closest candidate in the frontier is further away 
        than the furthest node in the current result set `W`.

        Args:
            q (ArrayLike): The query vector.
            ep (int): The entry point node ID for this layer.
            ef (int): Beam width (size of the dynamic candidate list).
            level (int): The graph layer level.
            logger (callable, optional): Callback for search path visualization.

        Returns:
            list[int]: The closest neighbor indices found, unsorted.
        """
        visited = set([ep])

        # Priority queues initialization
        # candidates: (distance, node_id) -> Min-Heap
        candidates = [(self.distance(self.data[ep], q), ep)]
        # W: (-distance, node_id) -> Max-Heap (using negative distance)
        W = [(-candidates[0][0], ep)] 

        if logger:
            logger({
                'event': 'init_search_layer',
                'layer': level,
                'ep': ep
            })

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            f_dist, f_id = W[0]

            # Early termination: if the closest candidate is worse than the worst 
            # node in our current top-ef list, we can't improve further.
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

                    # If e is better than the worst in W, or W is not full yet
                    if e_dist < -f_dist or len(W) < ef:
                        heapq.heappush(candidates, (e_dist, e_id))
                        heapq.heappush(W, (-e_dist, e_id))

                        if logger:
                            logger({
                                'event': 'accept_neighbor',
                                'current_node': c_id,
                                'neighbor': e_id
                            })

                        # Keep W size fixed at ef
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
        Performs K-Nearest Neighbors (KNN) search using the HNSW index.

        Traverses the graph from the top layer down to 0. At each layer, it finds 
        the local nearest neighbor to serve as the entry point for the next layer.
        At Layer 0, a broader search (`ef` size) is performed to find the top K results.

        Args:
            q (ArrayLike): The query/target vector.
            K (int, optional): The number of nearest neighbors to retrieve. Defaults to 1.
            logger (callable, optional): Callback for visualization logging. Defaults to None.

        Returns:
            list[int]: Indices of the K nearest neighbors, sorted by distance.
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

        # Coarse Search: Zoom in from top layer to layer 1
        for level in range(L, 0, -1):
            if logger and level != L:
                logger({
                    'event': 'layer_transition',
                    'from_layer': level + 1,
                    'to_layer': level,
                    'ep': ep
                })
            # ef=1 implies greedy search for the single best entry point
            W = self.search_layer(q, ep, 1, level, logger=logger)
            ep = W[0]

        if logger:
            logger({
                'event': 'layer_transition',
                'from_layer': 1,
                'to_layer': 0,
                'ep': ep
            })

        # Fine Search: Perform detailed search at layer 0
        W = self.search_layer(q, ep, max(K, self.ef), 0, logger=logger)

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
        Wrapper to select the best neighbors for a node.
        Defaults to the Heuristic strategy which balances proximity and spatial diversity.
        """
        # proba = self.rng.uniform(0.0, 1.0)
        # if proba <= 0.5:
        #     return self.select_neighbors_simple(q_id, W, M)
        return self.select_neighbors_heuristic(q_id, W, M, level)

    def select_neighbors_simple(self,
                                # q: ArrayLike,
                                q_id: int,
                                C: list[int],
                                M: int) -> tuple[list[int], list[float]]:
        """
        Selects neighbors using a naive strategy (Pure Proximity).

        Simply picks the `M` closest candidates by distance. 
        Note: This method is generally inferior as it can lead to clustering 
        and poor graph navigability (Long paths).

        Args:
            q_id (int): The node ID.
            C (list[int]): Candidate neighbors.
            M (int): Number of neighbors to return.

        Returns:
            tuple[list[int], list[float]]: Tuple of (indices, distances).
        """
        candidates = []
        for c in C:
            dist = self.distance(self.data[c], self.data[q_id])
            candidates.append((dist, c))
        
        nearest = heapq.nsmallest(M, candidates)
        
        return [n[1] for n in nearest], [n[0] for n in nearest]

    def select_neighbors_heuristic(self,
                                #    q: ArrayLike,
                                   q_id: int,
                                   C: list[int],
                                   M: int,
                                   level: int,
                                   extend_candidates: bool = True,
                                   keep_pruned_connections: bool = True) -> tuple[list[int], list[float]]:
        """
        Selects neighbors using the HNSW heuristic (Relative Neighborhood Graph approximation).

        This ensures spatial diversity by strictly accepting a neighbor `e` ONLY if 
        it is closer to the query `q` than to any already selected neighbor `r`.
        This prevents redundant connections in the same direction and maintains 
        the "Small World" navigability.

        Args:
            q_id (int): The ID of the query node.
            C (list[int]): Candidate neighbor IDs.
            M (int): Max number of neighbors to select.
            level (int): Graph level context.
            extend_candidates (bool, optional): If True, extends the candidate set by including
                neighbors of the candidates (defaults to True).
            keep_pruned_connections (bool, optional): If True, fills remaining slots in `M`
                with closest disregarded candidates to guarantee connectivity (defaults to True).

        Returns:
            tuple[list[int], list[float]]: Selected neighbor IDs and their distances.
        """
        R = []
        W = set(C)
        W_queue = []
        # Populate initial queue
        for c in C:
            heapq.heappush(W_queue, (self.distance(self.data[c], self.data[q_id]), c))

        # Heuristic: Extend candidates with their neighbors to bridge disconnected components
        if extend_candidates:
            temp_W = W.copy()
            for e in temp_W:
                for en_id, en_dist in self.graph[level][e].items():
                    if en_id not in W and en_id != q_id:
                        W.add(en_id)
                        heapq.heappush(W_queue, (self.distance(self.data[en_id], self.data[q_id]), en_id))

        W_d = [] # Queue for discarded candidates

        while len(W_queue) > 0 and len(R) < M:
            e_dist, e_id = heapq.heappop(W_queue)

            if e_id in W:
                W.remove(e_id)

            # Diversity Check:
            # Add e only if it is closer to q than to any node already in R.
            # This logic approximates the Relative Neighborhood Graph.
            is_good_neighbor = True
            for r_dist, r_id in R:
                dist_e_r = self.distance(self.data[e_id], self.data[r_id])
                if dist_e_r < e_dist:
                    is_good_neighbor = False
                    break

            if is_good_neighbor:
                R.append((e_dist, e_id))
            else:
                heapq.heappush(W_d, (e_dist, e_id))

        # Fallback: If we haven't filled M connections, use the closest discarded ones
        # to ensure the node isn't isolated.
        if keep_pruned_connections:
            while len(W_d) > 0 and len(R) < M:
                heapq.heappush(R, heapq.heappop(W_d))

        return [r[1] for r in R], [r[0] for r in R]

    # ===================================================
    # =============== VISUALIZE METHODS =================
    # ===================================================
    # Note: These methods rely on heavy external libraries (Plotly, NetworkX, Dash)
    # and are strictly used for debugging, education, or analysis of small graphs.

    def visualize_layer(self, l: int):
        """
        Visualizes the topology of a specific graph layer using NetworkX and Matplotlib.

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
        Sequentially visualizes multiple layers of the HNSW graph.

        Args:
            layers (list[int] | None, optional): Specific levels to plot.
                If None, visualizes all layers from 0 to max_level.
        """
        if layers is None:
            layers = range(self.max_level + 1)

        plt.ion()
        for l in layers:
            self.visualize_layer(l)
        plt.ioff()
        plt.show(block=True)

    def _compute_layout(self):
        """
        Computes the consistent 3D layout for visualization.
        
        Uses Layer 0 connectivity (spring layout) to determine X, Y coordinates,
        and uses the layer index for the Z coordinate.
        """
        G0 = nx.Graph()
        if self.graph:
            for node_id in self.graph[0]:
                G0.add_node(node_id)
                for neighbor_id in self.graph[0][node_id]:
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

    def _get_layer_color(self, level: int) -> str:
        """Returns a color string based on the layer level (Darker blue for higher levels)."""
        cmap = plt.get_cmap('Blues')
        intensity = 0.4 + 0.5 * (level / self.max_level) if self.max_level > 0 else 0.6
        rgba = cmap(intensity)
        return f'rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})'

    def _get_node_coords(self, node_positions, items, node_distances=None):
        """Extracts coordinate arrays and hover text for a set of nodes."""
        xs, ys, zs, items_text = [], [], [], []
        for node_id, l in items:
            if (node_id, l) in node_positions:
                x, y, z = node_positions[(node_id, l)]
                xs.append(x); ys.append(y); zs.append(z)
                dist_info = ""
                if node_distances:
                     dist = node_distances.get(node_id, float('inf'))
                     dist_info = f"<br>Dist: {dist:.4f}"
                items_text.append(f"ID: {node_id}<br>Layer: {l}{dist_info}")
        return xs, ys, zs, items_text

    def _get_edge_coords(self, node_positions, items):
        """Extracts line segments (including None separators) for a set of edges."""
        xs, ys, zs = [], [], []
        for (u, v), l in items:
            if (u, l) in node_positions and (v, l) in node_positions:
                p1, p2 = node_positions[(u, l)], node_positions[(v, l)]
                xs.extend([p1[0], p2[0], None]); ys.extend([p1[1], p2[1], None]); zs.extend([p1[2], p2[2], None])
        return xs, ys, zs

    def _get_vedge_coords(self, node_positions, items):
        """Extracts line segments for vertical edges (between layers)."""
        xs, ys, zs = [], [], []
        for (u1, l1), (u2, l2) in items:
            if (u1, l1) in node_positions and (u2, l2) in node_positions:
                p1, p2 = node_positions[(u1, l1)], node_positions[(u2, l2)]
                xs.extend([p1[0], p2[0], None]); ys.extend([p1[1], p2[1], None]); zs.extend([p1[2], p2[2], None])
        return xs, ys, zs

    def _reconstruct_path_coords(self, node_positions, active_nodes, parent_map):
        """Reconstructs the search path visualization from parent pointers."""
        path_x, path_y, path_z = [], [], []
        visited_edges = set()
        for node in active_nodes:
            curr = node
            while curr in parent_map:
                parent = parent_map[curr]
                edge_sig = tuple(sorted((curr, parent)))
                if edge_sig not in visited_edges:
                    visited_edges.add(edge_sig)
                    if curr in node_positions and parent in node_positions:
                        cx, cy, cz = node_positions[curr]
                        px, py, pz = node_positions[parent]
                        path_x.extend([px, cx, None]); path_y.extend([py, cy, None]); path_z.extend([pz, cz, None])
                curr = parent
        return path_x, path_y, path_z

    def _create_base_traces(self, node_positions, edge_opacity=1.0, plane_opacity=0.1, node_opacity=1.0, node_distances=None):
        """Generates the static Plotly traces (planes, static edges, nodes)."""
        x_edges, y_edges, z_edges = [], [], []
        x_vedges, y_vedges, z_vedges = [], [], []

        xs = [pos[0] for pos in node_positions.values()]
        ys = [pos[1] for pos in node_positions.values()]
        
        # Bounding Box
        padding = 0.1
        if xs and ys:
            x_min, x_max = min(xs) - padding, max(xs) + padding
            y_min, y_max = min(ys) - padding, max(ys) + padding
        else:
            x_min, x_max, y_min, y_max = -1, 1, -1, 1

        all_node_ids = set()
        if self.graph:
            all_node_ids = set(self.graph[0].keys())

        # Edges
        for l in range(self.max_level + 1):
            for node_id in all_node_ids:
                if (node_id, l) in node_positions:
                    x, y, z = node_positions[(node_id, l)]
                    if node_id in self.graph[l]:
                        for neighbor_id in self.graph[l][node_id]:
                            if node_id < neighbor_id:
                                if (neighbor_id, l) in node_positions:
                                    xn, yn, zn = node_positions[(neighbor_id, l)]
                                    x_edges.extend([x, xn, None]); y_edges.extend([y, yn, None]); z_edges.extend([z, zn, None])
                    if l < self.max_level and (node_id, l+1) in node_positions:
                        x_vedges.extend([x, x, None]); y_vedges.extend([y, y, None]); z_vedges.extend([l, l+1, None])

        traces: list[Any] = [
            go.Scatter3d(x=x_edges, y=y_edges, z=z_edges, mode='lines', line=dict(color='#cccccc', width=1), opacity=edge_opacity, hoverinfo='none', name='Edges'),
            go.Scatter3d(x=x_vedges, y=y_vedges, z=z_vedges, mode='lines', line=dict(color='#cccccc', width=1, dash='dash'), opacity=edge_opacity, hoverinfo='none', name='Vertical Edges ')
        ]

        
        for l in range(self.max_level + 1):
            color_hex = self._get_layer_color(l)
            
            
            # Plane
            traces.append(go.Surface(
                x=[[x_min, x_max], [x_min, x_max]], y=[[y_min, y_min], [y_max, y_max]], z=[[l, l], [l, l]],
                opacity=plane_opacity, showscale=False, colorscale=[[0, color_hex], [1, color_hex]], hoverinfo='none', name=f'Plane L{l}'
            ))

            # Nodes
            x_nodes, y_nodes, z_nodes, text_nodes = [], [], [], []
            for node_id in all_node_ids:
                # Logic: Only skip Entry Point if it's at the highest level (to draw it specially later)
                # If Entry Point is at a lower level, draw normally.
                if node_id == self.entry_point and l == self.max_level: 
                    continue

                if (node_id, l) in node_positions:
                    x, y, z = node_positions[(node_id, l)]
                    x_nodes.append(x); y_nodes.append(y); z_nodes.append(z)
                    
                    dist_info = ""
                    if node_distances and node_id in node_distances:
                        dist_info = f"<br>Dist: {node_distances[node_id]:.4f}"
                    text_nodes.append(f"ID: {node_id}<br>Layer: {l}{dist_info}")
            
            traces.append(go.Scatter3d(
                x=x_nodes, y=y_nodes, z=z_nodes, mode='markers',
                marker=dict(size=5, color=color_hex, line=dict(color='#333', width=0.5), opacity=node_opacity),
                text=text_nodes, hoverinfo='text', name=f'Nodes L{l}'
            ))

        # --- Entry Point (Drawn separately only at MAX LEVEL) ---
        if self.entry_point != -1:
            if (self.entry_point, self.max_level) in node_positions:
                x, y, z = node_positions[(self.entry_point, self.max_level)]
                
                dist_info = ""
                if node_distances and self.entry_point in node_distances:
                    dist_info = f"<br>Dist: {node_distances[self.entry_point]:.4f}"
                ep_text = [f"Entry Point ID: {self.entry_point}<br>Layer: {self.max_level}{dist_info}"]

                traces.append(go.Scatter3d(
                    x=[x], y=[y], z=[z], mode='markers',
                    # Characteristic DeepPink color
                    marker=dict(symbol='circle', size=6, color='deeppink', line=dict(color='white', width=1), opacity=node_opacity),
                    name='Entry Point Base', text=ep_text, hoverinfo='text'
                ))

        return traces

    def visualize_hierarchical_graph(self):
        """
        Launches a Dash application to interactively visualize the HNSW graph structure.
        Features hover effects to highlight nodes across all layers.
        """
        # Create layout and traces
        node_positions = self._compute_layout()
        base_traces = self._create_base_traces(node_positions)

        # Modify base traces to disable hover on nodes
        for trace in base_traces:
            if trace.name in ['Nodes', 'Entry Point']:
                trace.hoverinfo = 'skip'

        # Create Invisible Hitboxes for interaction
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
                aspectratio=dict(x=1, y=1, z=1)
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
        Creates a frame-by-frame 3D animation of the KNN search process.

        Visualizes:
        - The current candidate node being visited.
        - Accepted vs Rejected edges/nodes.
        - The "Frontier" (Min-heap) and "Best Results" (Max-heap).
        - The final path taken through layers.

        Args:
            q (ArrayLike): Query vector.
            k (int): Number of neighbors to find.
        
        Returns:
            list[int]: The resulting neighbor IDs (for verification).
        """
        q = np.array(q)
        search_log = []
        def logger(event): search_log.append(event)
        result_ids = self.knn_search(q, k, logger=logger)

        # 1. Distances & Layout
        node_distances = {}
        for idx, vec in enumerate(self.data):
            node_distances[idx] = self.distance(vec, q)

        node_positions = self._compute_layout()
        
        # 2. Base Traces
        base_traces = self._create_base_traces(
            node_positions, 
            edge_opacity=0.3, plane_opacity=0.05, node_opacity=0.1, node_distances=node_distances
        )

        # --- Trace Factory ---
        def create_traces(
            sel_e_x, sel_e_y, sel_e_z, sel_ve_x, sel_ve_y, sel_ve_z, 
            rej_e_x, rej_e_y, rej_e_z, con_e_x, con_e_y, con_e_z, 
            sel_n_x, sel_n_y, sel_n_z, sel_n_text, 
            rej_n_x, rej_n_y, rej_n_z, rej_n_text, 
            con_n_x, con_n_y, con_n_z, con_n_text, 
            cur_x, cur_y, cur_z, cur_text, 
            path_x, path_y, path_z, 
            blue_path_x, blue_path_y, blue_path_z,
            visited_history, 
            best_nodes_accumulated, 
            is_final_frame=False,
            final_result_ids=None
        ):
            traces = list(base_traces)
            
            # 1. Explored Nodes
            layer_groups = {l: [] for l in range(self.max_level + 1)}
            ep_active_coords = [] 

            if visited_history:
                for nid, lvl in visited_history:
                    # Logic Entry Point
                    if nid == self.entry_point and lvl == self.max_level:
                        if (nid, lvl) in node_positions:
                            ep_x, ep_y, ep_z = node_positions[(nid, lvl)]
                            dist = node_distances.get(nid, float('inf'))
                            ep_active_coords.append((ep_x, ep_y, ep_z, f"Entry Point ID: {nid}<br>Layer: {lvl}<br>Dist: {dist:.4f}"))
                        continue 
                    
                    if (nid, lvl) in best_nodes_accumulated:
                        continue

                    if lvl in layer_groups:
                        layer_groups[lvl].append((nid, lvl))
            
            for l in range(self.max_level + 1):
                vx, vy, vz, vt = [], [], [], []
                if layer_groups[l]:
                     vx, vy, vz, vt = self._get_node_coords(node_positions, layer_groups[l], node_distances)
                color = self._get_layer_color(l)
                traces.append(go.Scatter3d(
                    x=vx, y=vy, z=vz, mode='markers',
                    marker=dict(size=5, color=color, symbol='circle', line=dict(color='#333', width=0.5), opacity=1.0),
                    text=vt, hoverinfo='text', name=f'Explored L{l}'
                ))

            # 2. Persistent Best Nodes
            best_x, best_y, best_z, best_t = self._get_node_coords(node_positions, best_nodes_accumulated, node_distances)
            if best_x:
                traces.append(go.Scatter3d(
                    x=best_x, y=best_y, z=best_z, mode='markers',
                    marker=dict(size=6, color='green', symbol='circle', line=dict(color='black', width=1), opacity=1.0),
                    text=best_t, hoverinfo='text', name='Best of Layers'
                ))

            # 3. Active Entry Point
            ep_x, ep_y, ep_z, ep_t = [], [], [], []
            if ep_active_coords:
                 ep_x = [item[0] for item in ep_active_coords]
                 ep_y = [item[1] for item in ep_active_coords]
                 ep_z = [item[2] for item in ep_active_coords]
                 ep_t = [item[3] for item in ep_active_coords]

            traces.append(go.Scatter3d(
                x=ep_x, y=ep_y, z=ep_z, mode='markers',
                marker=dict(size=8, color='deeppink', symbol='circle', line=dict(color='white', width=1.5), opacity=1.0),
                text=ep_t, hoverinfo='text', name='Entry Point Active'
            ))

            # 4. Paths
            traces.append(go.Scatter3d(x=blue_path_x, y=blue_path_y, z=blue_path_z, mode='lines', line=dict(color='royalblue', width=3), opacity=0.4, name='Explored Path', hoverinfo='none'))
            traces.append(go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines', line=dict(color='#FF8C00', width=6), opacity=1.0, name='Search Path', hoverinfo='none'))
            
            # 5. Dynamic Search States
            if not is_final_frame:
                traces.extend([
                    go.Scatter3d(x=sel_e_x, y=sel_e_y, z=sel_e_z, mode='lines', line=dict(color='green', width=4), name='Select Edge'),
                    go.Scatter3d(x=sel_ve_x, y=sel_ve_y, z=sel_ve_z, mode='lines', line=dict(color='green', width=4, dash='dash'), name='Select V-Edge'),
                    go.Scatter3d(x=rej_e_x, y=rej_e_y, z=rej_e_z, mode='lines', line=dict(color='red', width=3), name='Reject Edge'),
                    go.Scatter3d(x=con_e_x, y=con_e_y, z=con_e_z, mode='lines', line=dict(color='yellow', width=3), name='Consider Edge'),
                    
                    go.Scatter3d(x=sel_n_x, y=sel_n_y, z=sel_n_z, mode='markers', marker=dict(size=6, color='green', symbol='circle', line=dict(color='black', width=0.5)), text=sel_n_text, hoverinfo='text', name='Select Node'),
                    go.Scatter3d(x=rej_n_x, y=rej_n_y, z=rej_n_z, mode='markers', marker=dict(size=6, color='red', symbol='circle', line=dict(color='black', width=0.5)), text=rej_n_text, hoverinfo='text', name='Reject Node'),
                    go.Scatter3d(x=con_n_x, y=con_n_y, z=con_n_z, mode='markers', marker=dict(size=6, color='yellow', symbol='circle', line=dict(color='black', width=0.5)), text=con_n_text, hoverinfo='text', name='Consider Node'),
                ])
                
                # Current Focus
                if cur_x:
                    traces.append(go.Scatter3d(
                        x=cur_x, y=cur_y, z=cur_z, mode='markers',
                        marker=dict(size=12, color='green', symbol='circle', opacity=0.5, line=dict(width=0)), 
                        hoverinfo='skip', name='Current Focus Halo'
                    ))
                    traces.append(go.Scatter3d(
                        x=cur_x, y=cur_y, z=cur_z, mode='markers',
                        marker=dict(size=5, color='green', symbol='circle', line=dict(color='black', width=1), opacity=1.0),
                        text=cur_text, hoverinfo='text', name='Current Focus'
                    ))
                else:
                    traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder Halo'))
                    traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder Node'))
                    
            else:
                for _ in range(9): traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder'))

            # 6. Top-K Results
            if is_final_frame and final_result_ids:
                res_items = [(rid, 0) for rid in final_result_ids]
                rx, ry, rz, rt = self._get_node_coords(node_positions, res_items, node_distances)
                if rx:
                    traces.append(go.Scatter3d(
                        x=rx, y=ry, z=rz, mode='markers', 
                        marker=dict(size=12, color='yellow', symbol='circle', opacity=0.5, line=dict(width=0)), 
                        text=rt, hoverinfo='text', hoverlabel=dict(bgcolor='green'), name='Result Halo'
                    ))
                    traces.append(go.Scatter3d(
                        x=rx, y=ry, z=rz, mode='markers', 
                        marker=dict(size=6, color='green', symbol='circle', line=dict(color='black', width=1)), 
                        hoverinfo='skip', name='Top-K Results'
                    ))
            else:
                traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder Halo'))
                traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder TopK'))

            return traces

        # --- Start Animation Frames ---
        frames = []
        
        # 0. Initial Blank Frame (Step 0)
        empty_dynamic_args = [[] for _ in range(28)]
        frames.append(go.Frame(data=create_traces(
            *empty_dynamic_args,
            [], [], [], [], [], [],
            set(), set()
        ), name='step_0'))

        # Loop
        current_node = -1; current_layer = -1
        current_W_set = set(); node_parents = {}; visited_history = set()
        best_nodes_accumulated = set()

        for i, log in enumerate(search_log):
            event = log['event']
            current_selected_edge = set(); current_selected_vedge = set()
            current_considered_edge = set(); current_rejected_edge = set()
            current_considered_node = set(); current_rejected_node = set()

            if event == 'init_knn_search':
                current_node = log['entry_point']; current_layer = log['max_level']
                current_W_set = {(current_node, current_layer)}
                visited_history.add((current_node, current_layer))
            elif event == 'layer_transition':
                best_nodes_accumulated.update(current_W_set)
                u = log['ep']; l_from = log['from_layer']; l_to = log['to_layer']
                node_parents[(u, l_to)] = (u, l_from)
                current_selected_vedge.add(((u, l_from), (u, l_to)))
                current_layer = l_to; current_node = u
                current_W_set = {(current_node, current_layer)}
                visited_history.add((current_node, current_layer))
            elif event == 'init_search_layer':
                current_layer = log['layer']; current_node = log['ep']
                current_W_set = {(current_node, current_layer)}
                visited_history.add((current_node, current_layer))
            elif event == 'visit_node':
                current_node = log['current_node']
                visited_history.add((current_node, current_layer))
            elif event == 'consider_neighbor':
                neighbor = log['neighbor']; u, v = sorted((current_node, neighbor))
                current_considered_edge.add(((u, v), current_layer))
                current_considered_node.add((neighbor, current_layer))
                if (neighbor, current_layer) not in node_parents:
                    node_parents[(neighbor, current_layer)] = (current_node, current_layer)
                visited_history.add((neighbor, current_layer))
            elif event == 'accept_neighbor':
                neighbor = log['neighbor']; u, v = sorted((current_node, neighbor))
                current_selected_edge.add(((u, v), current_layer))
                current_W_set.add((neighbor, current_layer))
            elif event == 'reject_neighbor':
                neighbor = log['neighbor']; u, v = sorted((current_node, neighbor))
                current_rejected_node.add((neighbor, current_layer))
                current_rejected_edge.add(((u, v), current_layer))
            elif event == 'reject_node':
                reject_node = log['reject_node']
                current_rejected_node.add((reject_node, current_layer))
                current_W_set.discard((reject_node, current_layer))
            elif event == 'reject_nodes':
                for id in log['reject_nodes']:
                    current_rejected_node.add((id, current_layer))
                    current_W_set.discard((id, current_layer))

            # Coords
            sel_e_x, sel_e_y, sel_e_z = self._get_edge_coords(node_positions, current_selected_edge)
            sel_ve_x, sel_ve_y, sel_ve_z = self._get_vedge_coords(node_positions, current_selected_vedge)
            rej_e_x, rej_e_y, rej_e_z = self._get_edge_coords(node_positions, current_rejected_edge)
            con_e_x, con_e_y, con_e_z = self._get_edge_coords(node_positions, current_considered_edge)
            sel_n_x, sel_n_y, sel_n_z, sel_n_text = self._get_node_coords(node_positions, current_W_set, node_distances)
            rej_n_x, rej_n_y, rej_n_z, rej_n_text = self._get_node_coords(node_positions, current_rejected_node, node_distances)
            con_n_x, con_n_y, con_n_z, con_n_text = self._get_node_coords(node_positions, current_considered_node, node_distances)
            cur_x, cur_y, cur_z, cur_text = self._get_node_coords(node_positions, [(current_node, current_layer)], node_distances)

            active_targets = current_W_set.copy(); active_targets.add((current_node, current_layer))
            path_x, path_y, path_z = self._reconstruct_path_coords(node_positions, active_targets, node_parents)
            blue_targets = visited_history.copy()
            blue_path_x, blue_path_y, blue_path_z = self._reconstruct_path_coords(node_positions, blue_targets, node_parents)

            frames.append(go.Frame(data=create_traces(
                sel_e_x, sel_e_y, sel_e_z, sel_ve_x, sel_ve_y, sel_ve_z,
                rej_e_x, rej_e_y, rej_e_z, con_e_x, con_e_y, con_e_z,
                sel_n_x, sel_n_y, sel_n_z, sel_n_text,
                rej_n_x, rej_n_y, rej_n_z, rej_n_text,
                con_n_x, con_n_y, con_n_z, con_n_text,
                cur_x, cur_y, cur_z, cur_text,
                path_x, path_y, path_z, blue_path_x, blue_path_y, blue_path_z,
                visited_history,
                best_nodes_accumulated.copy()
            ), name=f'step_{i+1}')) # Name shifted by 1

        # Final Frame
        final_targets = set([(rid, 0) for rid in result_ids if (rid, 0) in node_positions])
        # Final Frame
        final_targets = set([(rid, 0) for rid in result_ids if (rid, 0) in node_positions])
        final_path_x, final_path_y, final_path_z = self._reconstruct_path_coords(node_positions, final_targets, node_parents)
        final_blue_x, final_blue_y, final_blue_z = self._reconstruct_path_coords(node_positions, visited_history, node_parents)
        best_nodes_accumulated.update(current_W_set)

        frames.append(go.Frame(data=create_traces(
            *empty_dynamic_args,
            final_path_x, final_path_y, final_path_z,
            final_blue_x, final_blue_y, final_blue_z,
            visited_history,
            best_nodes_accumulated,
            is_final_frame=True,
            final_result_ids=result_ids
        ), name='final'))

        # Sliders
        sliders = [{
            "active": 0, "yanchor": "top", "xanchor": "left",
            "currentvalue": {"font": {"size": 20}, "prefix": "Step: ", "visible": True, "xanchor": "right"},
            "transition": {"duration": 0}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0,
            "steps": []
        }]

        for i, frame in enumerate(frames):
            step = {
                "args": [[frame.name], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
                "label": str(i) if frame.name != 'final' else 'Final', "method": "animate"
            }
            sliders[0]["steps"].append(step)

        # Init Figure (With Blank State)
        init_traces = create_traces(
            *empty_dynamic_args, 
            [], [], [], [], [], [],
            set(), set()
        )

        fig = go.Figure(
            data=init_traces,
            layout=go.Layout(
                title=f'HNSW Search Visualization (k={k})',
                scene=dict(
                    xaxis=dict(visible=False), yaxis=dict(visible=False), 
                    zaxis=dict(title='Level', showbackground=False, nticks=self.max_level + 2),
                    aspectmode='manual', aspectratio=dict(x=1, y=1, z=1) 
                ),
                uirevision='constant', hovermode='closest', sliders=sliders,
                updatemenus=[{'type': 'buttons', 'showactive': False, 'buttons': [
                    {'label': 'Play', 'method': 'animate', 'args': [None, {
                        'frame': {'duration': 250, 'redraw': True}, 
                        'fromcurrent': True, 'transition': {'duration': 250, 'easing': 'quadratic-in-out'}
                    }]},
                    {'label': 'Pause', 'method': 'animate', 'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False}, 
                        'mode': 'immediate', 'transition': {'duration': 0}
                    }]}
                ]}]
            ),
            frames=frames
        )
        fig.show()
        return result_ids

if __name__ == '__main__':
    train_data = np.random.rand(50, 100)
    query_data = np.random.rand(3, 100)

    max_elements = 50
    dim = 100
    M = 3
    ef_construction = 6

    index = HNSW('l2', dim)
    index.init_index(max_elements, M, ef_construction)

    index.insert_items(train_data)

    W = index.knn_search(query_data[0], 10)
    dists = [l2_distance(query_data[0], train_data[w])**2 for w in W]