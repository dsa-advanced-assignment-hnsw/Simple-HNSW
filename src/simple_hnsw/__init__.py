"""
Simple HNSW Implementation
==========================

A Python implementation of the Hierarchical Navigable Small World (HNSW) algorithm
for approximate nearest neighbor search.

This package provides:
- HNSW index construction and search.
- Support for L2 (Euclidean) and Cosine distance metrics.
- Visualization tools for the HNSW graph and search process.

Modules
-------
- hnsw: Core HNSW implementation.
- distance_metrics: Distance metric functions.
"""

from .hnsw import HNSW
from .distance_metrics import l2_distance, cosine_distance

__all__ = ['HNSW', 'l2_distance', 'cosine_distance']