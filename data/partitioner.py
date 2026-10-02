"""
Federated-learning dataset partitioners.

Includes:
    - RandomPartitioner: shuffled, approximately equal-size IID partition.
    - DirichletPartitioner: class-aware non-IID partition with retry, so every
      client is guaranteed a minimum number of samples of every class.

Both use local NumPy RNG instances and never touch global random state.
Pass only the TRAIN labels; returned indices refer to positions in that array.
"""

from typing import Any

import numpy as np


def _validate_partition(
    client_indices: dict[int, list[int]],
    num_samples: int,
    num_clients: int,
) -> None:
    """Verify that the partition is complete and non-overlapping."""
    if set(client_indices) != set(range(num_clients)):
        raise ValueError("Partition has incorrect client IDs")

    all_indices = [i for idx in client_indices.values() for i in idx]

    if len(all_indices) != num_samples:
        raise ValueError(
            "Partition does not contain every sample exactly once: "
            f"{len(all_indices)} assignments for {num_samples} samples."
        )
    if len(set(all_indices)) != num_samples:
        raise ValueError("Partition contains duplicate sample indices.")
    if any(i < 0 or i >= num_samples for i in all_indices):
        raise ValueError("Partition contains out-of-range sample indices.")


def _check_labels(labels: np.ndarray, num_clients: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D array")
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
    if len(labels) < num_clients:
        raise ValueError(
            f"Cannot create {num_clients} non-empty clients from only {len(labels)} samples."
        )
    return labels


class RandomPartitioner:
    """Random (IID) approximately equal-size partition."""

    def __init__(self, num_clients: int = 5, seed: int = 42):
        if num_clients < 1:
            raise ValueError("num_clients must be >= 1")
        self.num_clients = num_clients
        self.seed = seed

    def partition(self, labels: np.ndarray) -> tuple[dict[int, list[int]], dict[str, Any]]:
        labels = _check_labels(labels, self.num_clients)
        total_samples = len(labels)

        rng = np.random.default_rng(self.seed)
        splits = np.array_split(rng.permutation(total_samples), self.num_clients)

        client_indices = {cid: s.astype(int).tolist() for cid, s in enumerate(splits)}
        _validate_partition(client_indices, total_samples, self.num_clients)

        metadata = {
            "partition_method": "random",
            "num_clients": self.num_clients,
            "seed": self.seed,
            "num_samples": total_samples,
            "client_sizes": {str(c): len(i) for c, i in client_indices.items()},
        }
        return client_indices, metadata


# Deprecated alias for old configs/imports. Remove once nothing references it.
GeographicPartitioner = RandomPartitioner


class DirichletPartitioner:
    """
    Class-aware non-IID Dirichlet partitioner.

    For each class, samples are split among clients via Dirichlet(alpha).
    Smaller alpha -> stronger label skew.

    The draw is repeated (same seeded RNG, so still deterministic per seed)
    until every client has at least `min_samples_per_client` samples AND at
    least `min_class_samples_per_client` samples of every class. Use a value
    like 10 for a rare-positive task such as fraud so no client trains with
    zero positives.
    """

    def __init__(
        self,
        num_clients: int = 5,
        alpha: float = 0.5,
        seed: int = 42,
        min_samples_per_client: int = 10,
        min_class_samples_per_client: int = 2,
        max_tries: int = 1000,
    ):
        if num_clients < 1:
            raise ValueError("num_clients must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if min_samples_per_client < 1:
            raise ValueError("min_samples_per_client must be >= 1")
        if min_class_samples_per_client < 0:
            raise ValueError("min_class_samples_per_client must be >= 0")
        if max_tries < 1:
            raise ValueError("max_tries must be >= 1")

        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        self.min_samples_per_client = min_samples_per_client
        self.min_class_samples_per_client = min_class_samples_per_client
        self.max_tries = max_tries

    def _draw(
        self, labels: np.ndarray, classes: np.ndarray, rng: np.random.Generator
    ) -> dict[int, list[int]]:
        client_indices: dict[int, list[int]] = {c: [] for c in range(self.num_clients)}

        for cls in classes:
            class_indices = rng.permutation(np.flatnonzero(labels == cls))
            n = len(class_indices)

            proportions = rng.dirichlet(np.full(self.num_clients, self.alpha, dtype=float))
            raw = proportions * n
            counts = np.floor(raw).astype(int)

            remainder = n - int(counts.sum())
            if remainder > 0:
                order = np.argsort(-(raw - counts), kind="stable")
                counts[order[:remainder]] += 1

            start = 0
            for cid, count in enumerate(counts):
                end = start + int(count)
                client_indices[cid].extend(class_indices[start:end].astype(int).tolist())
                start = end

        return client_indices

    def _acceptable(
        self,
        labels: np.ndarray,
        classes: np.ndarray,
        client_indices: dict[int, list[int]],
    ) -> bool:
        for idx in client_indices.values():
            if len(idx) < self.min_samples_per_client:
                return False
            client_labels = labels[idx]
            for cls in classes:
                if np.count_nonzero(client_labels == cls) < self.min_class_samples_per_client:
                    return False
        return True

    def partition(self, labels: np.ndarray) -> tuple[dict[int, list[int]], dict[str, Any]]:
        labels = _check_labels(labels, self.num_clients)
        total_samples = len(labels)
        classes = np.unique(labels)

        class_counts = {str(c): int(np.count_nonzero(labels == c)) for c in classes}

        # One RNG advanced across attempts: deterministic per seed.
        rng = np.random.default_rng(self.seed)

        for attempt in range(1, self.max_tries + 1):
            client_indices = self._draw(labels, classes, rng)
            if self._acceptable(labels, classes, client_indices):
                break
        else:
            raise ValueError(
                f"No valid Dirichlet partition after {self.max_tries} tries "
                f"(num_clients={self.num_clients}, alpha={self.alpha}, "
                f"min_class_samples_per_client={self.min_class_samples_per_client}). "
                "Increase alpha, lower the minimums, or use fewer clients."
            )

        # Shuffle each client's sample order.
        for cid in range(self.num_clients):
            client_indices[cid] = rng.permutation(client_indices[cid]).astype(int).tolist()

        _validate_partition(client_indices, total_samples, self.num_clients)

        client_class_counts = {
            str(cid): {str(c): int(np.count_nonzero(labels[idx] == c)) for c in classes}
            for cid, idx in client_indices.items()
        }

        metadata = {
            "partition_method": "dirichlet",
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "attempts": attempt,
            "min_samples_per_client": self.min_samples_per_client,
            "min_class_samples_per_client": self.min_class_samples_per_client,
            "num_samples": total_samples,
            "num_classes": len(classes),
            "classes": [c.item() if hasattr(c, "item") else c for c in classes],
            "class_counts": class_counts,
            "client_sizes": {str(c): len(i) for c, i in client_indices.items()},
            "client_class_counts": client_class_counts,
        }
        return client_indices, metadata
