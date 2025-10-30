import numpy as np
from numpy.typing import ArrayLike

def l2_distance(a: ArrayLike, b: ArrayLike) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.linalg.norm(a - b))

def cosine_distance(a: ArrayLike, b: ArrayLike) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 1.0

    cosine_similarity = np.dot(a, b) / (norm_a * norm_b)
    cosine_distance = 1 - cosine_similarity

    return cosine_distance