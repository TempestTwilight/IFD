"""
Minimal smoke test demonstrating the new configuration system.

Run this to verify end-to-end reproducibility:
    uv run python examples/demo_config.py
"""

from config import RNGManager, load_config, save_provenance
from data.loader import load_ieee_cis_data
from data.partitioner import RandomPartitioner


def run_experiment(seed: int, output_dir: str):
    """Run a minimal experiment with the given seed."""
    # Load configuration with override
    config = load_config(
        overrides={
            "seed": seed,
            "data": {
                "num_clients": 5,
                "synthetic_fallback": True,
                "nrows": 500,
            },
        }
    )

    # Initialize RNG manager
    rng = RNGManager(config.seed)

    print(f"\n=== Experiment with seed={seed} ===")
    print(f"Master seed: {config.seed}")
    print(f"Partition seed: {rng.get_seed('partition')}")
    print(f"Model init seed: {rng.get_seed('model_init')}")

    # Load data (synthetic for demo)
    train_ds, test_ds = load_ieee_cis_data(
        data_dir="nonexistent",
        synthetic_fallback=True,
        nrows=config.data.nrows,
    )
    print(f"Loaded {len(train_ds)} train samples, {len(test_ds)} test samples")

    # Partition data
    partitioner = RandomPartitioner(
        num_clients=config.data.num_clients,
        seed=rng.get_seed("partition"),
    )
    client_indices, metadata = partitioner.partition(train_ds.y.numpy())

    print(f"Partitioned into {len(client_indices)} clients")
    for cid, indices in list(client_indices.items())[:3]:
        print(f"  Client {cid}: {len(indices)} samples")

    # Simulate metrics
    metrics = {
        "train_loss": 0.35,
        "test_auc": 0.87,
        "partition_method": metadata["partition_method"],
    }

    # Save provenance
    save_provenance(config, output_dir, metrics=metrics)
    print(f"Saved provenance to {output_dir}/")

    return client_indices


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Configuration System Demo")
    print("=" * 60)

    # Run with seed 42 twice
    print("\n--- Running experiment twice with seed=42 ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        indices1 = run_experiment(42, f"{tmpdir}/run1")
        indices2 = run_experiment(42, f"{tmpdir}/run2")

        # Verify reproducibility
        if indices1 == indices2:
            print("\n✓ REPRODUCIBLE: Same seed produces identical partitions")
        else:
            print("\n✗ ERROR: Same seed produced different partitions!")

    # Run with different seed
    print("\n--- Running experiment with seed=99 ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        indices3 = run_experiment(99, f"{tmpdir}/run3")

        # Verify difference
        if indices1 != indices3:
            print("\n✓ STOCHASTIC: Different seed produces different partitions")
        else:
            print("\n✗ ERROR: Different seeds produced identical partitions!")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
