"""Test configuration system."""

import tempfile
from pathlib import Path

from config import RNGManager, load_config, save_provenance


def test_default_config():
    """Test loading default configuration."""
    config = load_config()
    assert config.seed == 44
    assert config.data.num_clients == 10
    assert config.training.batch_size == 512
    assert config.fl.num_rounds == 20
    print("✓ Default config loads correctly")


def test_config_override():
    """Test CLI-style overrides."""
    overrides = {
        "seed": 99,
        "data": {"num_clients": 20},
        "training": {"batch_size": 256},
    }
    config = load_config(overrides=overrides)
    assert config.seed == 99
    assert config.data.num_clients == 20
    assert config.training.batch_size == 256
    assert config.fl.num_rounds == 20  # unchanged
    print("✓ Config overrides work correctly")


def test_rng_manager_determinism():
    """Test RNG manager produces deterministic streams."""
    rng1 = RNGManager(42)
    rng2 = RNGManager(42)

    # Same master seed → same derived seeds
    assert rng1.get_seed("partition") == rng2.get_seed("partition")
    assert rng1.get_seed("model_init") == rng2.get_seed("model_init")

    # Different streams → different seeds
    assert rng1.get_seed("partition") != rng1.get_seed("model_init")

    print("✓ RNG streams are deterministic and independent")


def test_rng_numpy_generators():
    """Test numpy generators produce reproducible sequences."""
    rng1 = RNGManager(42)
    rng2 = RNGManager(42)

    gen1 = rng1.get_numpy_rng("partition")
    gen2 = rng2.get_numpy_rng("partition")

    vals1 = gen1.random(10)
    vals2 = gen2.random(10)

    assert (vals1 == vals2).all()
    print("✓ NumPy generators are reproducible")


def test_rng_different_seeds():
    """Test different master seeds produce different streams."""
    rng1 = RNGManager(42)
    rng2 = RNGManager(99)

    assert rng1.get_seed("partition") != rng2.get_seed("partition")
    print("✓ Different master seeds produce different streams")


def test_rng_state_capture_restore():
    """Test RNG state capture and restoration."""
    import random

    import numpy as np
    import torch

    rng = RNGManager(42)

    # Generate some random numbers
    random.random()
    np.random.rand()
    torch.rand(1)

    # Capture state
    state = rng.get_rng_state()

    # Generate more numbers
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    # Restore state
    rng.set_rng_state(state)

    # Should get same sequence
    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2

    print("✓ RNG state capture/restore works")


def test_provenance_saving():
    """Test provenance metadata is saved."""
    config = load_config(overrides={"seed": 123})

    with tempfile.TemporaryDirectory() as tmpdir:
        save_provenance(config, tmpdir, metrics={"auc": 0.95})

        # Check files exist
        assert (Path(tmpdir) / "config.json").exists()
        assert (Path(tmpdir) / "manifest.json").exists()

        # Verify content
        import json

        with open(Path(tmpdir) / "config.json") as f:
            saved_config = json.load(f)
        assert saved_config["seed"] == 123

        with open(Path(tmpdir) / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["seed"] == 123
        assert "timestamp" in manifest
        assert "git_commit" in manifest
        assert manifest["metrics"]["auc"] == 0.95

    print("✓ Provenance saving works")


def test_config_serialization():
    """Test config can be serialized to dict."""
    config = load_config()
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict["seed"] == 44
    assert config_dict["data"]["num_clients"] == 10
    print("✓ Config serialization works")


if __name__ == "__main__":
    test_default_config()
    test_config_override()
    test_rng_manager_determinism()
    test_rng_numpy_generators()
    test_rng_different_seeds()
    test_rng_state_capture_restore()
    test_provenance_saving()
    test_config_serialization()
    print("\n✓ All configuration tests passed")
