import numpy as np

from numpy.typing import ArrayLike, NDArray
from simple_hnsw.distance_metrics import l2_distance, cosine_distance

def brute_force_search(train_dataset: ArrayLike,
                       query_vectors: ArrayLike,
                       K: int = 1) -> tuple[NDArray, NDArray]:
    train_dataset = np.atleast_2d(train_dataset)
    query_vectors = np.atleast_2d(query_vectors)
    
    indices = []
    distances = []

    for query_vector in query_vectors:
        # Calculate the Euclidean distance from the current query vector to all vectors in the train dataset
        distance = np.linalg.norm(query_vector - train_dataset, axis=1)

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

    indices, distances = brute_force_search(train_dataset, query_vectors)

    print("\nSorted Indices for each query vector:\n", indices)
    print("\n" + "="*40)
    print("\nSorted Distances for each query vector:\n", distances)