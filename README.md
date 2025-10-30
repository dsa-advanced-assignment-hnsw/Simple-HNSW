
# Simple-HNSW

A compact, educational implementation of the HNSW (Hierarchical Navigable Small World) approximate nearest neighbor algorithm in Python, alongside a brute-force baseline for comparison and testing.

This repository is intended for learning and experimentation. It provides a small, readable HNSW implementation in `src/simple_hnsw` and a simple brute-force search implementation in `src/brute_force` so you can compare accuracy and performance.

## Repository layout

- `src/simple_hnsw/`
	- `hnsw.py` — a minimal, easy-to-read HNSW implementation (insert, knn search, neighbor selection, etc.).
	- `distance_metrics.py` — distance functions used by the index (L2 and cosine).
- `src/brute_force/`
	- `brute_force_search.py` — straightforward KNN by computing distances to every point.
- `tests/` — simple unit tests (run with `pytest`).

## Features

- HNSW index supporting L2 and cosine distance.
- Insert many vectors (`insert_items`) and run K-NN queries (`knn_search`).
- Brute-force baseline for correctness and benchmarking.

## Requirements

- Python 3.8+
- See `requirements.txt` for exact dependencies. Install with:

```bash
python -m pip install -r requirements.txt
```

## Quick usage

Below are short examples that demonstrate how to use the provided modules.

### Brute-force search

```python
import numpy as np
from src.brute_force.brute_force_search import brute_force_search

train = np.random.rand(1000, 128)
queries = np.random.rand(5, 128)
K = 5

indices, distances = brute_force_search(train, queries, K=K)
print(indices.shape)   # (num_queries, K)
print(distances.shape) # (num_queries, K)
```

### Simple HNSW index

```python
import numpy as np
from src.simple_hnsw.hnsw import HNSW

dim = 128
index = HNSW('l2', dim)
index.init_index(max_elements=1000, M=8, ef_construction=100, random_seed=42)

train = np.random.rand(500, dim)
index.insert_items(train)

q = np.random.rand(dim)
neighbors = index.knn_search(q, K=5)
print('neighbor ids:', neighbors)
```

Notes:
- The implementation is intentionally small and educational. It does not aim to be production-grade or highly optimized.
- `init_index` must be called before inserting data.

## Tests

Run the test suite with `pytest` from the project root:

```bash
pytest -q
```

There is a small test in `tests/test_hnsw.py` to check basic behavior.

## Development notes

- The HNSW implementation in `src/simple_hnsw/hnsw.py` contains a `__main__` section demonstrating a simple usage example; you can run it directly for a quick smoke test.
- For experimentation, tweak parameters like `M`, `ef_construction`, and `ef` to see their effect on accuracy and graph construction.

## Contributing

Contributions are welcome. For small fixes or documentation updates, open a pull request with a short description of your change.

If you add features or refactor code, please:
- Add tests for new or changed behavior
- Keep the code readable and well-commented (this project is meant to teach the algorithm)

## License

No LICENSE file is included in the repository. If you plan to publish or share this project, consider adding a license (for example, MIT) to make reuse intentions explicit.

## Contact / References

- Implementation inspired by the original HNSW paper by Malkov & Yashunin. Use the paper as a reference for deeper understanding.