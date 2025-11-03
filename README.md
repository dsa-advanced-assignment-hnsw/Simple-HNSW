
# Simple-HNSW

A Python implementation of the HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search, with performance benchmarking against the established `hnswlib` library.

This project provides a clean, educational implementation of HNSW in Python, along with comparative benchmarking tools to evaluate its performance against the industry-standard `hnswlib` implementation.

## Repository Layout

- `src/simple_hnsw/`
  - `hnsw.py` — Core HNSW implementation with insert and kNN search functionality
  - `distance_metrics.py` — Implementation of distance metrics (L2)
- `src/brute_force/`
  - `brute_force_search.py` — Baseline brute-force search implementation
- `tests/`
  - `test_hnsw.py` — Benchmarking and comparison with `hnswlib`

## Features

- Clean HNSW implementation optimized for readability and learning
- L2 distance metric support
- Comparative benchmarking with `hnswlib`
- Brute-force baseline implementation
- Performance measurement tools

## Requirements

- Python 3.8+
- NumPy
- hnswlib (for benchmarking)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Usage

### Simple HNSW Implementation

```python
from simple_hnsw import hnsw
import numpy as np

# Initialize index
dim = 100
index = hnsw.HNSW('l2', dim)
index.init_index(max_elements=10000, M=16, ef_construction=32)

# Add data
data = np.random.rand(10000, dim)
index.insert_items(data)

# Search
query = np.random.rand(dim)
neighbors = index.knn_search(query, k=20)
```

### Comparison with hnswlib

```python
import hnswlib

# Initialize hnswlib index
index = hnswlib.Index('l2', dim)
index.init_index(max_elements=10000, M=16, ef_construction=32)

# Add data
index.add_items(data)

# Search
neighbors, distances = index.knn_query(query, k=20)
```

## Parameters

Key parameters that affect performance and accuracy:

- `dim`: Dimensionality of the vectors
- `M`: Maximum number of connections per element (default: 16)
- `ef_construction`: Size of the dynamic candidate list during construction (default: 32)
- `max_elements`: Maximum number of elements that can be stored in the index

## Benchmarking

Run the benchmark comparison:
```bash
python tests/test_hnsw.py
```

This will:
1. Generate random test data (10,000 vectors, 100 dimensions)
2. Build indices using both implementations
3. Perform k-NN searches
4. Compare build times and search times
5. Verify search results

## Performance Tips

1. **Parameter Tuning**
   - Increase `M` for better accuracy at the cost of memory and build time
   - Higher `ef_construction` improves graph quality but increases build time
   - For search, larger `ef` values give better accuracy but slower search

2. **Memory Usage**
   - Memory scales with `max_elements * M`
   - Keep dimensionality (`dim`) and `M` reasonable for your use case

## Contributing

Contributions are welcome! Areas for improvement:
- Additional distance metrics
- Performance optimizations
- Extended benchmarking tools
- Visualization utilities

Please ensure:
- Code remains readable and well-documented
- Include tests for new features
- Benchmark results for performance changes

## References

1. Original HNSW paper: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Yu. A. Malkov and D. A. Yashunin
2. `hnswlib`: https://github.com/nmslib/hnswlib

## License

[MIT License]