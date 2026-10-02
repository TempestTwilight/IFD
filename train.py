"""
Training entrypoint for IFD-PART2.

Features:
    - Reproducible training via --seed.
    - Explicit Ray shutdown.
    - Atomic metrics JSON writes.
    - Optional final model checkpoint.
    - Supports CascadeRouter and baseline strategies.
    - Supports L1/L2/L3 ablations.
"""

import argparse
import json
import os
import sys
import time

# WSL2 workaround. Proper Ray cleanup is still performed.
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

import flwr as fl
import ray
import torch
from torch.utils.data import DataLoader, Subset

from config import RNGManager, load_config, save_provenance
from config.cli import args_to_overrides
from data.loader import load_ieee_cis_data
from data.partitioner import GeographicPartitioner
from experiment.client import (
    FraudMLP,
    IFDClient,
    set_parameters,
)
from orchestration.flower_strategy import CascadeRouter

# ============================================================================
# Argument parsing
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="IFD-PART2 Federated Training")

    p.add_argument(
        "--data-dir",
        default="./data/raw",
    )

    p.add_argument(
        "--nrows",
        type=int,
        default=None,
    )

    p.add_argument(
        "--num-clients",
        type=int,
        default=10,
    )

    p.add_argument(
        "--num-rounds",
        type=int,
        default=20,
    )

    p.add_argument(
        "--num-adversaries",
        type=int,
        default=0,
    )

    p.add_argument(
        "--attack-type",
        type=str,
        default=None,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    p.add_argument(
        "--epochs-per-round",
        type=int,
        default=1,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=44,
    )

    p.add_argument(
        "--run-label",
        type=str,
        default=None,
    )

    p.add_argument(
        "--save-model",
        type=str,
        default=None,
    )

    p.add_argument(
        "--results-dir",
        type=str,
        default="./results",
    )

    p.add_argument(
        "--disable-layer1",
        action="store_true",
    )

    p.add_argument(
        "--disable-layer2",
        action="store_true",
    )

    p.add_argument(
        "--disable-layer3",
        action="store_true",
    )

    p.add_argument(
        "--baseline",
        type=str,
        default=None,
    )

    return p.parse_args()


# ============================================================================
# Ray cleanup
# ============================================================================


def shutdown_ray():
    """
    Shut down the local Ray runtime.

    Safe to call if Ray was never initialized or was already shut down.
    """

    try:
        if ray.is_initialized():
            print(
                "\n[Ray cleanup] Shutting down Ray...",
                flush=True,
            )

            ray.shutdown(
                _exiting_interpreter=False,
                wait_for_processes=True,
            )

            print(
                "[Ray cleanup] Ray shutdown complete.",
                flush=True,
            )

        else:
            print(
                "\n[Ray cleanup] Ray is not initialized.",
                flush=True,
            )

    except Exception as exc:
        print(
            f"[Ray cleanup] WARNING: {type(exc).__name__}: {exc}",
            flush=True,
        )


# ============================================================================
# Atomic JSON writer
# ============================================================================


def atomic_json_dump(data, path):
    """
    Write JSON atomically.

    The temporary file prevents an interrupted process from leaving behind
    a partially-written result that could be mistaken for a valid result.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = path + ".tmp"

    try:
        with open(
            tmp_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                default=str,
            )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp_path,
            path,
        )

    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

        raise


# ============================================================================
# Main
# ============================================================================


def main():

    args = parse_args()
    config = load_config(overrides=args_to_overrides(args))
    rng = RNGManager(config.seed)

    # ------------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------------
    # (RNGManager handles np.random.seed, torch.manual_seed, and torch.cuda.manual_seed_all)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    if config.data.num_clients < 1:
        raise ValueError("--num-clients must be >= 1")

    if config.fl.num_rounds < 1:
        raise ValueError("--num-rounds must be >= 1")

    if config.attack.num_adversaries < 0:
        raise ValueError("--num-adversaries must be >= 0")

    if config.attack.num_adversaries > config.data.num_clients:
        raise ValueError("--num-adversaries cannot exceed --num-clients")

    if config.data.nrows is not None and config.data.nrows < 1:
        raise ValueError("--nrows must be >= 1")

    if config.training.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if config.training.epochs_per_round < 1:
        raise ValueError("--epochs-per-round must be >= 1")

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Device: {device}",
        flush=True,
    )

    if device.type == "cuda":
        print(
            f"  GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            flush=True,
        )

    # Ray actors already provide process isolation.
    num_workers = 0
    pin_memory = False

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    print(
        f"\nLoading data from: {config.data.data_dir}",
        flush=True,
    )

    try:
        train_ds, test_ds = load_ieee_cis_data(
            data_dir=config.data.data_dir,
            nrows=config.data.nrows,
            synthetic_fallback=False,
        )

    except FileNotFoundError:
        print(
            f"ERROR: IEEE-CIS CSV files not found in '{config.data.data_dir}'.",
            flush=True,
        )

        return 1

    print(
        f"  Train samples: {len(train_ds)}",
        flush=True,
    )

    print(
        f"  Test samples:  {len(test_ds)}",
        flush=True,
    )

    input_dim = train_ds[0][0].shape[0]

    print(
        f"  Input dim: {input_dim}",
        flush=True,
    )

    # ------------------------------------------------------------------------
    # Partition data
    # ------------------------------------------------------------------------

    # Check for sweep-level partition file (created by run_seed_sweep.py)
    partition_file = os.path.join(os.path.dirname(config.output.results_dir), "partition.json")

    if os.path.exists(partition_file):
        print(
            f"Loading sweep-level partition from {partition_file}",
            flush=True,
        )

        with open(partition_file) as f:
            partition_data = json.load(f)

        client_indices = {int(k): v for k, v in partition_data["client_indices"].items()}

        print(
            f"Partition config: {partition_data['config']}",
            flush=True,
        )

        print(
            f"Samples per client: {[len(v) for v in client_indices.values()]}",
            flush=True,
        )

    else:
        # Fallback: create partition per-run (for non-sweep usage)
        print(
            "No sweep-level partition found, creating partition per-run",
            flush=True,
        )

        labels = train_ds.y.numpy()

        partitioner = GeographicPartitioner(
            num_clients=config.data.num_clients,
            seed=rng.get_seed("partition"),
        )

        client_indices, _ = partitioner.partition(labels)

    # ------------------------------------------------------------------------
    # Flower client factory
    # ------------------------------------------------------------------------

    def client_fn(cid: str):

        client_idx = int(cid)

        indices = client_indices[client_idx]

        client_train_ds = Subset(
            train_ds,
            indices,
        )

        train_loader = DataLoader(
            client_train_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            test_ds,
            batch_size=config.training.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        is_adv = client_idx < config.attack.num_adversaries

        client = IFDClient(
            cid=cid,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            epochs=config.training.epochs_per_round,
            lr=config.training.lr,
            is_adversary=is_adv,
            attack_type=(config.attack.attack_type if is_adv else None),
        )

        return client.to_client()

    # ------------------------------------------------------------------------
    # Metrics aggregation
    # ------------------------------------------------------------------------

    def weighted_average(metrics):

        if not metrics:
            return {}

        total_examples = sum(n for n, _ in metrics)

        if total_examples == 0:
            return {}

        keys = metrics[0][1].keys()

        return {key: sum(n * values[key] for n, values in metrics) / total_examples for key in keys}

    # ------------------------------------------------------------------------
    # Ablation layers
    # ------------------------------------------------------------------------

    from layers.layer1_norm_cosine import (
        Layer1NormCosine,
    )
    from layers.layer2_spectral import (
        Layer2Spectral,
    )
    from layers.layer3_temporal import (
        Layer3Temporal,
    )

    class _PassL1(Layer1NormCosine):
        def score(self, gradients):

            n = len(gradients)

            return (
                torch.ones(n),
                torch.ones(n),
            )

    class _PassL2(Layer2Spectral):
        def score(self, gradients):

            n = len(gradients)

            return (
                torch.ones(n),
                torch.ones(n),
            )

    class _PassL3(Layer3Temporal):
        def score(
            self,
            gradients,
            client_ids=None,
        ):

            n = len(gradients)

            return (
                torch.ones(n),
                torch.ones(n),
            )

    layer1 = _PassL1() if config.defense.disable_layer1 else None

    layer2 = _PassL2() if config.defense.disable_layer2 else None

    layer3 = _PassL3() if config.defense.disable_layer3 else None

    # ------------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------------

    if config.baseline.baseline:
        from orchestration.baseline_strategy import (
            BaselineStrategy,
        )

        strategy = BaselineStrategy(
            baseline_name=config.baseline.baseline,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        print(
            f"Strategy: BaselineStrategy ({config.baseline.baseline})",
            flush=True,
        )

    else:
        strategy = CascadeRouter(
            layer1=layer1,
            layer2=layer2,
            layer3=layer3,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        print(
            "Strategy: CascadeRouter",
            flush=True,
        )

    # ------------------------------------------------------------------------
    # Experiment information
    # ------------------------------------------------------------------------

    print(
        "\nStarting FL simulation:",
        flush=True,
    )

    print(
        f"  Clients:      {config.data.num_clients}",
        flush=True,
    )

    print(
        f"  Rounds:       {config.fl.num_rounds}",
        flush=True,
    )

    print(
        f"  Adversaries:  {config.attack.num_adversaries}",
        flush=True,
    )

    print(
        f"  Attack:       {config.attack.attack_type}",
        flush=True,
    )

    # ------------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------------

    t0 = time.time()

    history = None

    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config.data.num_clients,
            config=fl.server.ServerConfig(num_rounds=config.fl.num_rounds),
            strategy=strategy,
            client_resources={
                "num_cpus": 10,
                "num_gpus": 0.5,
            },
            ray_init_args={
                "ignore_reinit_error": True,
            },
        )

        elapsed = time.time() - t0

        print(
            f"\nTraining complete in "
            f"{elapsed:.1f}s "
            f"({elapsed / max(1, config.fl.num_rounds):.1f}s/round)",
            flush=True,
        )

    except Exception as exc:
        elapsed = time.time() - t0

        print(
            f"\nERROR: FL simulation failed after {elapsed:.1f}s",
            flush=True,
        )

        print(
            f"  {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise

    finally:
        shutdown_ray()

    # ------------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------------

    if history is None:
        print(
            "ERROR: Simulation returned no history.",
            flush=True,
        )

        return 1

    # ------------------------------------------------------------------------
    # Save metrics atomically
    # ------------------------------------------------------------------------

    os.makedirs(
        config.output.results_dir,
        exist_ok=True,
    )

    if config.output.run_label:
        filename = f"{config.output.run_label}.json"

    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        run_type = (
            f"attack_{config.attack.attack_type}_{config.attack.num_adversaries}adv"
            if config.attack.num_adversaries > 0
            else "clean"
        )

        filename = f"history_{run_type}_{timestamp}.json"

    metrics_path = os.path.join(
        config.output.results_dir,
        filename,
    )

    metrics = {
        "seed": config.seed,
        "run_label": config.output.run_label,
        "num_clients": config.data.num_clients,
        "num_rounds": config.fl.num_rounds,
        "num_adversaries": config.attack.num_adversaries,
        "attack_type": config.attack.attack_type,
        "losses_distributed": history.losses_distributed,
        "metrics_distributed": history.metrics_distributed,
        "metrics_centralized": history.metrics_centralized,
    }

    atomic_json_dump(
        metrics,
        metrics_path,
    )

    print(
        f"Metrics saved to: {metrics_path}",
        flush=True,
    )

    # ------------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------------

    provenance_dir = os.path.dirname(metrics_path) if metrics_path else config.output.results_dir
    save_provenance(config, provenance_dir, metrics=metrics)
    print(f"Provenance saved to: {provenance_dir}/config.json", flush=True)

    # ------------------------------------------------------------------------
    # Optional model checkpoint
    # ------------------------------------------------------------------------

    if config.output.save_model:
        save_dir = os.path.dirname(config.output.save_model)

        if save_dir:
            os.makedirs(
                save_dir,
                exist_ok=True,
            )

        final_model = FraudMLP(input_dim=input_dim).to(device)

        if (
            hasattr(
                strategy,
                "latest_aggregated_ndarrays",
            )
            and strategy.latest_aggregated_ndarrays
        ):
            set_parameters(
                final_model,
                strategy.latest_aggregated_ndarrays,
            )

            tmp_model = config.output.save_model + ".tmp"

            torch.save(
                final_model.state_dict(),
                tmp_model,
            )

            os.replace(
                tmp_model,
                config.output.save_model,
            )

            print(
                f"Model checkpoint saved to: {config.output.save_model}",
                flush=True,
            )

        else:
            print(
                "WARNING: No aggregated parameters available; model checkpoint not saved.",
                flush=True,
            )

    return 0


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted by user.",
            flush=True,
        )

        shutdown_ray()

        sys.exit(130)

    except Exception as exc:
        print(
            f"\nFATAL ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        shutdown_ray()

        sys.exit(1)
