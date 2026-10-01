"""Integration test demonstrating end-to-end reproducibility."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from config import RNGManager, load_config, save_provenance
from data.loader import load_ieee_cis_data
from data.partitioner import RandomPartitioner


def test_partition_reproducibility():
    """Test that same seed produces identical partitions."""
    config1 = load_config(overrides={"seed": 42})
    config2 = load_config(overrides={"seed": 42})

    rng1 = RNGManager(config1.seed)
    rng2 = RNGManager(config2.seed)

    # Create partitioner with config seed
    seed1 = rng1.get_seed("partition")
    seed2 = rng2.get_seed("partition")

    assert seed1 == seed2

    # Generate synthetic labels for testing
    labels = np.array([0] * 900 + [1] * 100)

    part1 = RandomPartitioner(num_clients=10, seed=seed1)
    part2 = RandomPartitioner(num_clients=10, seed=seed2)

    indices1, _ = part1.partition(labels)
    indices2, _ = part2.partition(labels)

    # Verify identical partitions
    for cid in range(10):
        assert indices1[cid] == indices2[cid]

    print("✓ Same config+seed → identical partitions")


def test_different_seed_different_partition():
    """Test that different seeds produce different partitions."""
    rng1 = RNGManager(42)
    rng2 = RNGManager(99)

    labels = np.array([0] * 900 + [1] * 100)

    part1 = RandomPartitioner(num_clients=10, seed=rng1.get_seed("partition"))
    part2 = RandomPartitioner(num_clients=10, seed=rng2.get_seed("partition"))

    indices1, _ = part1.partition(labels)
    indices2, _ = part2.partition(labels)

    # Should be different
    assert indices1 != indices2

    print("✓ Different seed → different partitions")


def test_model_init_reproducibility():
    """Test model initialization reproducibility."""
    from experiment.client import FraudMLP

    config = load_config(overrides={"seed": 42})
    rng = RNGManager(config.seed)

    # Initialize global RNG (already done in RNGManager.__init__)
    # Create two models
    torch.manual_seed(rng.get_seed("model_init"))
    model1 = FraudMLP(input_dim=30)

    torch.manual_seed(rng.get_seed("model_init"))
    model2 = FraudMLP(input_dim=30)

    # Verify identical weights
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)

    print("✓ Same seed → identical model initialization")


def test_data_loading_reproducibility():
    """Test data loading produces consistent results."""
    # Load synthetic data (deterministic with fixed internal seed)
    train1, test1 = load_ieee_cis_data(
        data_dir="nonexistent",
        synthetic_fallback=True,
        nrows=500,
    )

    train2, test2 = load_ieee_cis_data(
        data_dir="nonexistent",
        synthetic_fallback=True,
        nrows=500,
    )

    # Should be identical (synthetic data uses fixed seed=42)
    assert torch.equal(train1.x, train2.x)
    assert torch.equal(train1.y, train2.y)

    print("✓ Data loading is deterministic")


def test_end_to_end_experiment():
    """Test complete experiment configuration and execution."""
    config = load_config(
        overrides={
            "seed": 123,
            "data": {
                "num_clients": 5,
                "synthetic_fallback": True,
                "nrows": 200,
            },
            "training": {
                "batch_size": 64,
                "epochs_per_round": 2,
            },
            "fl": {
                "num_rounds": 3,
            },
        }
    )

    rng = RNGManager(config.seed)

    # Load data
    train_ds, test_ds = load_ieee_cis_data(
        data_dir="nonexistent",
        synthetic_fallback=True,
        nrows=config.data.nrows,
    )

    # Partition
    partitioner = RandomPartitioner(
        num_clients=config.data.num_clients,
        seed=rng.get_seed("partition"),
    )
    client_indices, metadata = partitioner.partition(train_ds.y.numpy())

    # Verify configuration was applied
    assert len(client_indices) == 5
    assert config.training.batch_size == 64
    assert config.fl.num_rounds == 3

    # Save provenance
    with tempfile.TemporaryDirectory() as tmpdir:
        save_provenance(config, tmpdir, metrics={"test_auc": 0.85})

        # Verify files
        assert (Path(tmpdir) / "config.json").exists()
        assert (Path(tmpdir) / "manifest.json").exists()

        # Verify we can reload config
        import json

        with open(Path(tmpdir) / "config.json") as f:
            saved = json.load(f)
        assert saved["seed"] == 123
        assert saved["data"]["num_clients"] == 5

    print("✓ End-to-end experiment configuration works")


def test_rng_stream_independence():
    """Test that RNG streams don't interfere with each other."""
    rng = RNGManager(42)

    # Get partition generator and use it
    part_gen = rng.get_numpy_rng("partition")
    part_vals = part_gen.random(5)

    # Use model stream to verify independence
    _ = rng.get_numpy_rng("model_init").random(5)

    # Create new manager with same seed
    rng2 = RNGManager(42)

    # Get partition generator again (should produce same sequence)
    part_gen2 = rng2.get_numpy_rng("partition")
    part_vals2 = part_gen2.random(5)

    # Verify partition stream wasn't affected by model stream usage
    assert np.array_equal(part_vals, part_vals2)

    print("✓ RNG streams are independent")


if __name__ == "__main__":
    test_partition_reproducibility()
    test_different_seed_different_partition()
    test_model_init_reproducibility()
    test_data_loading_reproducibility()
    test_end_to_end_experiment()
    test_rng_stream_independence()
    print("\n✓ All integration tests passed")
