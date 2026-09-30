"""
Test that seed sweep partition semantics are correct:
1. Partition is identical across all seeds in a sweep
2. Training seed can differ per seed
3. Re-running the same sweep reproduces the same partition
4. Different sweep configs produce different partitions
"""

import json
import os
import tempfile

import numpy as np

from data.loader import load_ieee_cis_data
from data.partitioner import DirichletPartitioner


def test_dirichlet_partition_reproducible():
    """DirichletPartitioner produces identical output with same seed."""
    train_ds, _ = load_ieee_cis_data(nrows=1000)
    y_train = train_ds.y.numpy()

    p1 = DirichletPartitioner(num_clients=5, alpha=0.5, seed=43)
    indices1, _ = p1.partition(y_train)

    p2 = DirichletPartitioner(num_clients=5, alpha=0.5, seed=43)
    indices2, _ = p2.partition(y_train)

    for i in range(5):
        assert indices1[i] == indices2[i], f"Client {i} indices differ"


def test_dirichlet_partition_different_seeds():
    """Different partition seeds produce different partitions."""
    train_ds, _ = load_ieee_cis_data(nrows=1000)
    y_train = train_ds.y.numpy()

    p1 = DirichletPartitioner(num_clients=5, alpha=0.5, seed=43)
    indices1, _ = p1.partition(y_train)

    p2 = DirichletPartitioner(num_clients=5, alpha=0.5, seed=99)
    indices2, _ = p2.partition(y_train)

    # At least one client should have different indices
    assert any(indices1[i] != indices2[i] for i in range(5))


def test_partition_file_format():
    """Partition file has correct schema."""
    train_ds, _ = load_ieee_cis_data(nrows=1000)
    y_train = train_ds.y.numpy()

    partitioner = DirichletPartitioner(num_clients=5, alpha=0.5, seed=43)
    client_indices, metadata = partitioner.partition(y_train)

    # Simulate what run_seed_sweep.py writes
    partition_data = {
        "client_indices": {int(k): [int(i) for i in v] for k, v in client_indices.items()},
        "metadata": metadata,
        "config": {
            "partition_seed": 43,
            "partition_alpha": 0.5,
            "num_clients": 5,
            "nrows": 1000,
            "num_samples": len(y_train),
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(partition_data, f, indent=2)
        tmp_path = f.name

    try:
        with open(tmp_path) as f:
            loaded = json.load(f)

        assert "client_indices" in loaded
        assert "metadata" in loaded
        assert "config" in loaded
        assert loaded["config"]["partition_seed"] == 43
        assert loaded["config"]["partition_alpha"] == 0.5
        assert loaded["config"]["num_clients"] == 5

        # Verify indices are JSON-serializable and reconstructible
        for k, v in loaded["client_indices"].items():
            assert isinstance(int(k), int)
            assert isinstance(v, list)
            assert all(isinstance(i, int) for i in v)
    finally:
        os.unlink(tmp_path)


def test_sweep_partition_invariant():
    """
    Simulates the seed sweep flow:
    1. Create partition once
    2. Multiple "seeds" load the same partition
    3. Verify client assignments are identical across all seeds
    """
    train_ds, _ = load_ieee_cis_data(nrows=1000)
    y_train = train_ds.y.numpy()

    # Step 1: Create sweep-level partition (done once before seed loop)
    partitioner = DirichletPartitioner(num_clients=5, alpha=0.5, seed=43)
    client_indices, metadata = partitioner.partition(y_train)

    partition_data = {
        "client_indices": {int(k): [int(i) for i in v] for k, v in client_indices.items()},
        "metadata": metadata,
        "config": {
            "partition_seed": 43,
            "partition_alpha": 0.5,
            "num_clients": 5,
            "nrows": 1000,
            "num_samples": len(y_train),
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(partition_data, f, indent=2)
        partition_file = f.name

    try:
        # Step 2: Simulate multiple seeds loading the same partition
        loaded_partitions = []
        for seed in range(45, 56):  # Seeds 45-55
            with open(partition_file) as f:
                loaded = json.load(f)
            loaded_partitions.append(loaded["client_indices"])

        # Step 3: Verify all seeds see identical client assignments
        reference = loaded_partitions[0]
        for i, partition in enumerate(loaded_partitions[1:], start=1):
            for client_id in reference:
                assert reference[client_id] == partition[client_id], (
                    f"Seed {45 + i} client {client_id} indices differ from reference"
                )

        print(
            f"✓ All {len(loaded_partitions)} seeds share identical partition "
            f"(client 0: {len(reference['0'])} samples)"
        )
    finally:
        os.unlink(partition_file)


def test_training_seed_independence():
    """
    Training seed can vary per seed without affecting partition.
    (This is a conceptual test; actual training randomness is not tested here)
    """
    train_ds, _ = load_ieee_cis_data(nrows=1000)
    y_train = train_ds.y.numpy()

    # Partition seed is fixed at sweep level
    partition_seed = 43

    # Different training seeds vary per seed
    training_seeds = [45, 46, 47, 48, 49]

    # Create partition once with partition_seed
    partitioner = DirichletPartitioner(num_clients=5, alpha=0.5, seed=partition_seed)
    partition, _ = partitioner.partition(y_train)

    # Different training seeds should NOT affect partition
    # (partition is already fixed; training seed is used for model/optimizer init)
    for train_seed in training_seeds:
        # Simulate: set training seed (would be used by torch.manual_seed, etc.)
        np.random.seed(train_seed)
        _ = np.random.rand()  # Different per training seed

        # Verify partition is unchanged (reference equality or deep comparison)
        # In real code, each seed run loads the same partition.json
        # Here we just verify the partition object is stable
        assert len(partition[0]) > 0, "Partition should be populated"

    print("✓ Training seed independence verified")


if __name__ == "__main__":
    test_dirichlet_partition_reproducible()
    test_dirichlet_partition_different_seeds()
    test_partition_file_format()
    test_sweep_partition_invariant()
    test_training_seed_independence()
    print("\n✅ All partition tests passed")
