# Simple-HNSW

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A pure Python implementation of the **Hierarchical Navigable Small World (HNSW)** algorithm for approximate nearest neighbor search. This project is designed primarily for **educational purposes**, offering a clean, readable codebase to understand the inner workings of HNSW, while also providing powerful **interactive 3D visualizations** to explore the graph structure and search process.

It includes benchmarking tools to compare performance against the industry-standard `hnswlib` library, allowing users to see the trade-offs between a pure Python implementation and optimized C++ bindings.

## ✨ Features

- **Pure Python Implementation**: Easy to read, debug, and modify. Ideal for learning the HNSW algorithm.
- **Distance Metrics**: Supports **L2 (Euclidean)** and **Cosine** distance metrics.
- **Interactive 3D Visualization**:
    - **Layer Visualization**: View individual graph layers using Matplotlib.
    - **Hierarchical Graph Explorer**: Interactive 3D dashboard using Dash & Plotly to explore the multi-layer graph structure.
    - **Search Process Visualization**: Watch the search algorithm navigate the graph in real-time 3D.
- **Benchmarking Suite**: Comprehensive tools to compare build time, search time, and recall against `hnswlib`.
- **Brute-Force Baseline**: Includes a brute-force search implementation for ground-truth accuracy verification.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Simple-HNSW.git
    cd Simple-HNSW
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

### Basic Example

Here's how to initialize the index, insert data, and perform a k-NN search.

```python
import numpy as np
from simple_hnsw.hnsw import HNSW

# 1. Generate random data
dim = 100
num_elements = 1000
data = np.random.rand(num_elements, dim)

# 2. Initialize HNSW index
# space: 'l2' or 'cosine'
index = HNSW(space='l2', dim=dim)
index.init_index(max_elements=num_elements, M=16, ef_construction=200)

# 3. Insert items
index.insert_items(data)

# 4. Search
query = np.random.rand(dim)
# k: number of nearest neighbors to find
neighbors = index.knn_search(query, k=10)

print("Nearest Neighbors Indices:", neighbors)
```

### 📊 Visualization

This project shines in its ability to visualize the HNSW graph structure.

#### 1. Visualize a Single Layer
View the connections in a specific layer (0 is the bottom layer).

```python
# Visualize layer 0
index.visualize_layer(0)
```

#### 2. Interactive Hierarchical Dashboard
Launch a Dash web application to explore the entire graph in 3D. You can see how nodes are distributed across layers and their connections.

```python
# Opens a local Dash server (usually http://127.0.0.1:8050/)
index.visualize_hierarchical_graph()
```

#### 3. Visualize Search Process
Trace the path of the search algorithm as it navigates from the top layer down to the target.

```python
# Visualize the search for a specific query
index.visualize_search(query, k=5)
```

## ⚡ Benchmarking

Compare the performance of `Simple-HNSW` against `hnswlib`.

```bash
python tests/test_hnsw.py
```

**What it does:**
1.  Generates random test data (default: 10,000 vectors, 100 dimensions).
2.  Builds indices using both `Simple-HNSW` and `hnswlib`.
3.  Performs k-NN searches.
4.  Reports:
    - **Build Time**: Time taken to construct the index.
    - **Search Time**: Average time per query.
    - **Recall**: Accuracy of the approximate search compared to ground truth.

## 📂 Project Structure

```
Simple-HNSW/
├── src/
│   ├── simple_hnsw/
│   │   ├── hnsw.py              # Core HNSW implementation & Visualization logic
│   │   └── distance_metrics.py  # L2 and Cosine distance functions
│   └── brute_force/
│       └── brute_force_search.py # Baseline for accuracy comparison
├── tests/
│   └── test_hnsw.py             # Benchmarking script
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## ⚙️ Key Parameters

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `M` | Max number of connections per element. Higher `M` = better recall but higher memory/build time. | `16` |
| `ef_construction` | Size of the dynamic candidate list during construction. Higher `ef` = better graph quality but slower build. | `200` |
| `ef` | Size of the dynamic candidate list during search. Higher `ef` = better recall but slower search. | `10` |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**References:**
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin)
- [hnswlib](https://github.com/nmslib/hnswlib)