"""
Geographic/Dirichlet partitioner for federated learning.
"""

from typing import Any

import numpy as np


class GeographicPartitioner:
    def __init__(self, num_clients: int = 5, seed: int = 42):
        self.num_clients = num_clients
        self.seed = seed

    def partition(self, labels: np.ndarray) -> tuple[dict[int, list[int]], dict[str, Any]]:
        """Partition label array into client index sets."""
        np.random.seed(self.seed)
        total_samples = len(labels)
        indices = np.random.permutation(total_samples)
        splits = np.array_split(indices, self.num_clients)
        client_indices = {i: splits[i].tolist() for i in range(self.num_clients)}
        return client_indices, {}
