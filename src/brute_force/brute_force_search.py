import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np

from typing import Literal
from numpy.typing import ArrayLike, NDArray
from simple_hnsw.distance_metrics import l2_distance, cosine_distance

def brute_force_search(space: Literal['l2', 'cosine'],
                       train_dataset: ArrayLike,
                       query_vectors: ArrayLike,
                       K: int = 1) -> tuple[NDArray, NDArray]:
    """
    Perform brute force search to find K nearest neighbors.

    Args:
        space (Literal['l2', 'cosine']): The distance metric to use ('l2' or 'cosine').
        train_dataset (ArrayLike): The dataset to search in.
        query_vectors (ArrayLike): The query vectors.
        K (int, optional): The number of nearest neighbors to return. Defaults to 1.

    Returns:
        tuple[NDArray, NDArray]: A tuple containing the indices and distances of the K nearest neighbors.
    """
    train_dataset = np.atleast_2d(train_dataset)
    query_vectors = np.atleast_2d(query_vectors)

    dist_func = l2_distance if space == 'l2' else cosine_distance

    indices = []
    distances = []

    for query_vector in query_vectors:
        # Calculate the distance from the current query vector to all vectors in the train dataset
        distance = [dist_func(query_vector, train_data) for train_data in train_dataset]
        distance = np.array(distance)

        # Get the sorted indices based on distance
        sorted_indices = np.argsort(distance)

        # Use the sorted indices to reorder both the distances and the original indices
        sorted_distance = distance[sorted_indices]
        sorted_index = sorted_indices

        # Append the sorted results to the lists
        distances.append(sorted_distance[:K])
        indices.append(sorted_index[:K])

    return np.array(indices), np.array(distances)

if __name__ == '__main__':
    train_dataset = np.random.rand(10, 5)
    print("Train Dataset:\n", train_dataset)

    query_vectors = np.random.rand(3, 5)
    print("Query vectors:\n", query_vectors)

    indices, distances = brute_force_search('l2', train_dataset, query_vectors, 3)

    print("\nSorted Indices for each query vector:\n", indices)
    print("\n" + "="*40)
    print("\nSorted Distances for each query vector:\n", distances)