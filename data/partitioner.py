"""
Federated-learning dataset partitioners.

Includes:
    - RandomPartitioner: shuffled, approximately equal-size IID partition.
    - DirichletPartitioner: class-aware non-IID partition.

The partitioners use local NumPy RNG instances so they do not modify
global NumPy random state. This is important for reproducible training
seed sweeps.
"""

from typing import Any

import numpy as np


class GeographicPartitioner:
    """Random approximately equal-size partition."""

    def __init__(self, num_clients: int = 5, seed: int = 42):
        if num_clients < 1:
            raise ValueError("num_clients must be >= 1")

        self.num_clients = num_clients
        self.seed = seed

    def partition(
        self,
        labels: np.ndarray,
    ) -> tuple[dict[int, list[int]], dict[str, Any]]:
        """
        Randomly partition samples into approximately equal-size clients.

        Each sample is assigned to exactly one client.
        """

        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError("labels must be a 1-D array")

        total_samples = len(labels)

        if total_samples < self.num_clients:
            raise ValueError(
                f"Cannot create {self.num_clients} non-empty clients "
                f"from only {total_samples} samples."
            )

        rng = np.random.default_rng(self.seed)

        indices = rng.permutation(total_samples)
        splits = np.array_split(indices, self.num_clients)

        client_indices = {
            client_id: split.astype(int).tolist() for client_id, split in enumerate(splits)
        }

        self._validate_partition(
            client_indices,
            total_samples,
            self.num_clients,
        )

        metadata = {
            "partition_method": "random",
            "num_clients": self.num_clients,
            "seed": self.seed,
            "num_samples": total_samples,
            "client_sizes": {
                str(client_id): len(indices) for client_id, indices in client_indices.items()
            },
        }

        return client_indices, metadata

    @staticmethod
    def _validate_partition(
        client_indices: dict[int, list[int]],
        num_samples: int,
        num_clients: int,
    ) -> None:
        expected_clients = set(range(num_clients))

        if set(client_indices) != expected_clients:
            raise ValueError("Partition has incorrect client IDs")

        all_indices = [index for indices in client_indices.values() for index in indices]

        if len(all_indices) != num_samples:
            raise ValueError("Partition does not contain every sample exactly once")

        if len(set(all_indices)) != num_samples:
            raise ValueError("Partition contains duplicate sample indices")

        if any(index < 0 or index >= num_samples for index in all_indices):
            raise ValueError("Partition contains out-of-range sample indices")


class DirichletPartitioner:
    """
    Class-aware non-IID Dirichlet partitioner.

    For each class, samples are distributed among clients according to
    a Dirichlet(alpha) distribution.

    Smaller alpha -> stronger non-IID behavior.
    Larger alpha -> more balanced class distributions.
    """

    def __init__(
        self,
        num_clients: int = 5,
        alpha: float = 0.5,
        seed: int = 42,
        min_samples_per_client: int = 1,
    ):
        if num_clients < 1:
            raise ValueError("num_clients must be >= 1")

        if alpha <= 0:
            raise ValueError("alpha must be > 0")

        if min_samples_per_client < 1:
            raise ValueError("min_samples_per_client must be >= 1")

        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        self.min_samples_per_client = min_samples_per_client

    def partition(
        self,
        labels: np.ndarray,
    ) -> tuple[dict[int, list[int]], dict[str, Any]]:
        """
        Partition samples using a class-aware Dirichlet distribution.

        The returned indices refer directly to positions in the supplied
        labels array. Every sample is assigned to exactly one client.
        """

        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError("labels must be a 1-D array")

        total_samples = len(labels)

        if total_samples < self.num_clients:
            raise ValueError(
                f"Cannot create {self.num_clients} clients from {total_samples} samples."
            )

        classes = np.unique(labels)

        if len(classes) == 0:
            raise ValueError("labels must not be empty")

        rng = np.random.default_rng(self.seed)

        client_indices: dict[int, list[int]] = {
            client_id: [] for client_id in range(self.num_clients)
        }

        class_counts: dict[str, int] = {}

        # ------------------------------------------------------------
        # Partition each class independently
        # ------------------------------------------------------------

        for cls in classes:
            class_indices = np.flatnonzero(labels == cls)

            class_counts[str(cls)] = len(class_indices)

            if len(class_indices) == 0:
                continue

            class_indices = rng.permutation(class_indices)

            proportions = rng.dirichlet(
                np.full(
                    self.num_clients,
                    self.alpha,
                    dtype=float,
                )
            )

            # Convert proportions into integer counts while ensuring
            # that the total count exactly equals the class size.
            counts = np.floor(proportions * len(class_indices)).astype(int)

            remainder = len(class_indices) - int(counts.sum())

            if remainder > 0:
                # Give leftover samples to clients with the largest
                # fractional proportions.
                fractional = (proportions * len(class_indices)) - counts

                order = np.argsort(
                    -fractional,
                    kind="stable",
                )

                for client_id in order[:remainder]:
                    counts[client_id] += 1

            start = 0

            for client_id, count in enumerate(counts):
                end = start + int(count)

                if end > start:
                    client_indices[client_id].extend(class_indices[start:end].astype(int).tolist())

                start = end

        # ------------------------------------------------------------
        # Shuffle each client's final sample order
        # ------------------------------------------------------------

        for client_id in range(self.num_clients):
            indices = client_indices[client_id]

            if indices:
                shuffled = rng.permutation(indices)
                client_indices[client_id] = shuffled.astype(int).tolist()

        # ------------------------------------------------------------
        # Validate complete partition
        # ------------------------------------------------------------

        self._validate_partition(
            client_indices,
            total_samples,
            self.num_clients,
        )

        client_sizes = {
            str(client_id): len(indices) for client_id, indices in client_indices.items()
        }

        empty_clients = [
            client_id
            for client_id, indices in client_indices.items()
            if len(indices) < self.min_samples_per_client
        ]

        if empty_clients:
            raise ValueError(
                "Dirichlet partition produced clients with fewer than "
                f"{self.min_samples_per_client} samples: "
                f"{empty_clients}. "
                "Increase alpha, increase the dataset size, or use "
                "a different partition seed."
            )

        # ------------------------------------------------------------
        # Per-client class counts
        # ------------------------------------------------------------

        client_class_counts: dict[str, dict[str, int]] = {}

        for client_id, indices in client_indices.items():
            client_labels = labels[indices]

            counts_for_client: dict[str, int] = {}

            for cls in classes:
                counts_for_client[str(cls)] = int(np.sum(client_labels == cls))

            client_class_counts[str(client_id)] = counts_for_client

        metadata = {
            "partition_method": "dirichlet",
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "num_samples": total_samples,
            "num_classes": len(classes),
            "classes": [cls.item() if hasattr(cls, "item") else cls for cls in classes],
            "class_counts": class_counts,
            "client_sizes": client_sizes,
            "client_class_counts": client_class_counts,
        }

        return client_indices, metadata

    @staticmethod
    def _validate_partition(
        client_indices: dict[int, list[int]],
        num_samples: int,
        num_clients: int,
    ) -> None:
        """Verify that the partition is complete and non-overlapping."""

        expected_clients = set(range(num_clients))

        if set(client_indices) != expected_clients:
            raise ValueError("Partition has incorrect client IDs")

        all_indices = [index for indices in client_indices.values() for index in indices]

        if len(all_indices) != num_samples:
            raise ValueError(
                "Partition does not contain every sample exactly once: "
                f"{len(all_indices)} assignments for "
                f"{num_samples} samples."
            )

        if len(set(all_indices)) != num_samples:
            raise ValueError("Partition contains duplicate sample indices.")

        if any(index < 0 or index >= num_samples for index in all_indices):
            raise ValueError("Partition contains out-of-range sample indices.")
