#!/usr/bin/env python3

"""
Seed sweep runner.

Runs run_experiments.py once per seed (45-55).

Features:
    - Seed-level resume.
    - Skips seeds with COMPLETE marker.
    - Per-seed stdout/stderr log.
    - Seed timeout.
    - Process-group termination.
    - Continues after failed/timed-out seeds.
    - Sweep-level summary.
    - Sweep-level data partitioning shared across all seeds.
    - Fixed Dirichlet partition generated from the loader's training dataset.

Usage:
    python run_seed_sweep.py

Optional environment variables:
    NUM_CLIENTS
    NUM_ROUNDS
    NROWS
    EXPERIMENT_TIMEOUT_MINUTES
    SEED_TIMEOUT_MINUTES
    PARTITION_SEED (default: 43)
    PARTITION_ALPHA (default: 0.5)
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime

import numpy as np

from data.loader import load_ieee_cis_data
from data.partitioner import DirichletPartitioner

# ============================================================================
# Configuration
# ============================================================================

SEEDS = list(range(45, 56))

BASE_RESULTS = "./results/Seed_Sweep"

SWEEP_LOG = os.path.join(
    BASE_RESULTS,
    "seed_sweep_log.txt",
)

SEED_TIMEOUT_MINUTES = float(
    os.environ.get(
        "SEED_TIMEOUT_MINUTES",
        "3600",
    )
)

SEED_TIMEOUT_SECONDS = SEED_TIMEOUT_MINUTES * 60

PARTITION_SEED = int(
    os.environ.get(
        "PARTITION_SEED",
        "43",
    )
)

PARTITION_ALPHA = float(
    os.environ.get(
        "PARTITION_ALPHA",
        "0.5",
    )
)

NUM_CLIENTS = int(
    os.environ.get(
        "NUM_CLIENTS",
        "5",
    )
)

NROWS = int(
    os.environ.get(
        "NROWS",
        "10000",
    )
)

PARTITION_FILE = os.path.join(
    BASE_RESULTS,
    "partition.json",
)


# ============================================================================
# Utility
# ============================================================================


def ts() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(
    message: str,
    fh=None,
) -> None:
    print(
        message,
        flush=True,
    )

    if fh:
        fh.write(message + "\n")
        fh.flush()


def seed_complete(results_dir: str) -> bool:
    return os.path.isfile(
        os.path.join(
            results_dir,
            "COMPLETE",
        )
    )


# ============================================================================
# Partition utilities
# ============================================================================


def create_partition(
    partition_file: str,
    sweep_log,
) -> None:
    """
    Create and save one fixed partition for the entire seed sweep.

    The partition is created from the training dataset returned by
    load_ieee_cis_data(). The test dataset is never partitioned.
    """

    log(
        "Creating sweep-level partition...",
        sweep_log,
    )

    log(
        f"  Partition seed: {PARTITION_SEED}",
        sweep_log,
    )

    log(
        f"  Partition alpha: {PARTITION_ALPHA}",
        sweep_log,
    )

    log(
        f"  Num clients: {NUM_CLIENTS}",
        sweep_log,
    )

    log(
        f"  Loading training data (nrows={NROWS})...",
        sweep_log,
    )

    train_dataset, test_dataset = load_ieee_cis_data(
        nrows=NROWS,
    )

    y_train = train_dataset.y.numpy()

    log(
        f"  Training samples: {len(train_dataset)}",
        sweep_log,
    )

    log(
        f"  Test samples: {len(test_dataset)}",
        sweep_log,
    )

    log(
        f"  Input dimension: {train_dataset.x.shape[1]}",
        sweep_log,
    )

    partitioner = DirichletPartitioner(
        num_clients=NUM_CLIENTS,
        alpha=PARTITION_ALPHA,
        seed=PARTITION_SEED,
    )

    client_indices, metadata = partitioner.partition(
        y_train,
    )

    # Convert NumPy integer values to regular Python integers
    # so that JSON serialization is reliable.
    client_indices_json = {
        client_id: [index for index in indices] for client_id, indices in client_indices.items()
    }

    partition_data = {
        "client_indices": client_indices_json,
        "metadata": metadata,
        "config": {
            "partition_seed": PARTITION_SEED,
            "partition_alpha": PARTITION_ALPHA,
            "num_clients": NUM_CLIENTS,
            "nrows": NROWS,
            "num_train_samples": len(train_dataset),
            "num_test_samples": len(test_dataset),
            "input_dim": train_dataset.x.shape[1],
        },
    }

    with open(
        partition_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            partition_data,
            f,
            indent=2,
        )

    log(
        f"  Saved partition to {partition_file}",
        sweep_log,
    )

    log(
        "  Samples per client: "
        + str({client_id: len(indices) for client_id, indices in client_indices.items()}),
        sweep_log,
    )

    # Useful for verifying the fraud distribution.
    log(
        "  Fraud samples per client: "
        + str(
            {
                client_id: int(y_train[np.asarray(indices, dtype=int)].sum())
                for client_id, indices in client_indices.items()
            }
        ),
        sweep_log,
    )


def load_partition(
    partition_file: str,
) -> dict:
    """Load a previously created sweep-level partition."""

    with open(
        partition_file,
        encoding="utf-8",
    ) as f:
        return json.load(f)


def validate_partition(
    partition_data: dict,
) -> None:
    """
    Verify that an existing partition matches the current configuration.

    This prevents accidentally reusing a partition generated with
    different clients, alpha, seed, or dataset size.
    """

    config = partition_data.get("config", {})

    expected = {
        "partition_seed": PARTITION_SEED,
        "partition_alpha": PARTITION_ALPHA,
        "num_clients": NUM_CLIENTS,
        "nrows": NROWS,
    }

    for key, expected_value in expected.items():
        actual_value = config.get(key)

        if actual_value != expected_value:
            raise ValueError(
                f"Existing partition mismatch for '{key}': "
                f"stored={actual_value!r}, "
                f"expected={expected_value!r}. "
                "Delete partition.json or restore the original configuration."
            )

    client_indices = partition_data.get(
        "client_indices",
        {},
    )

    if len(client_indices) != NUM_CLIENTS:
        raise ValueError(
            "Existing partition contains "
            f"{len(client_indices)} clients, "
            f"but NUM_CLIENTS={NUM_CLIENTS}."
        )


# ============================================================================
# Process termination
# ============================================================================


def terminate_process_tree(
    process,
    seed_log,
) -> None:
    if process.poll() is not None:
        return

    try:
        seed_log.write(f"\n[{ts()}] Terminating seed process tree.\n")
        seed_log.flush()

        if os.name == "posix":
            os.killpg(
                process.pid,
                signal.SIGTERM,
            )
        else:
            process.terminate()

        try:
            process.wait(
                timeout=15,
            )

        except subprocess.TimeoutExpired:
            seed_log.write(f"[{ts()}] Forcing seed process termination.\n")
            seed_log.flush()

            if os.name == "posix":
                os.killpg(
                    process.pid,
                    signal.SIGKILL,
                )
            else:
                process.kill()

            process.wait(
                timeout=15,
            )

    except ProcessLookupError:
        pass

    except Exception as exc:
        seed_log.write(f"[{ts()}] Termination error: {type(exc).__name__}: {exc}\n")
        seed_log.flush()


# ============================================================================
# Run one seed
# ============================================================================


def run_seed(
    seed: int,
    sweep_log,
) -> str:

    results_dir = os.path.join(
        BASE_RESULTS,
        f"seed_{seed}",
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------------

    if seed_complete(results_dir):
        message = f"Seed {seed:3d}: SKIPPED — already complete."

        log(
            message,
            sweep_log,
        )

        return message

    # ------------------------------------------------------------------------
    # Per-seed log
    # ------------------------------------------------------------------------

    seed_log_path = os.path.join(
        results_dir,
        "seed.log",
    )

    log(
        f"\n--- Seed {seed} ---",
        sweep_log,
    )

    log(
        f"Directory: {results_dir}",
        sweep_log,
    )

    log(
        f"Started: {ts()}",
        sweep_log,
    )

    start = time.time()

    try:
        with open(
            seed_log_path,
            "a",
            encoding="utf-8",
        ) as seed_log:
            seed_log.write(
                "\n"
                + "=" * 80
                + "\n"
                + f"SEED {seed} STARTED {ts()}\n"
                + f"TIMEOUT: {SEED_TIMEOUT_MINUTES:.1f} min\n"
                + f"PARTITION FILE: {PARTITION_FILE}\n"
                + f"PARTITION SEED: {PARTITION_SEED}\n"
                + f"PARTITION ALPHA: {PARTITION_ALPHA}\n"
                + "=" * 80
                + "\n"
            )

            seed_log.flush()

            env = os.environ.copy()

            # Training/output configuration.
            env["RESULTS_DIR"] = results_dir
            env["SEED"] = str(seed)

            # ----------------------------------------------------------------
            # Critical:
            # Explicitly tell run_experiments.py to use the fixed partition.
            # ----------------------------------------------------------------
            env["PARTITION_FILE"] = PARTITION_FILE

            # Also expose partition configuration explicitly.
            env["PARTITION_SEED"] = str(PARTITION_SEED)
            env["PARTITION_ALPHA"] = str(PARTITION_ALPHA)
            env["NUM_CLIENTS"] = str(NUM_CLIENTS)
            env["NROWS"] = str(NROWS)

            cmd = [
                sys.executable,
                "run_experiments.py",
            ]

            kwargs = {
                "env": env,
                "stdout": seed_log,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
            }

            if os.name == "posix":
                kwargs["start_new_session"] = True
            else:
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            process = subprocess.Popen(
                cmd,
                **kwargs,
            )

            try:
                return_code = process.wait(
                    timeout=SEED_TIMEOUT_SECONDS,
                )

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start

                message = f"Seed {seed:3d}: TIMEOUT — {elapsed / 60:.1f} min"

                seed_log.write(f"\n[{ts()}] {message}\n")
                seed_log.flush()

                terminate_process_tree(
                    process,
                    seed_log,
                )

                log(
                    message,
                    sweep_log,
                )

                return message

            elapsed = time.time() - start

            # ----------------------------------------------------------------
            # Verify seed completion marker.
            # ----------------------------------------------------------------

            if return_code == 0 and seed_complete(results_dir):
                message = f"Seed {seed:3d}: SUCCESS — {elapsed / 60:.1f} min — finished {ts()}"

            elif return_code == 0:
                message = (
                    f"Seed {seed:3d}: FAILED — "
                    "run_experiments.py returned 0 "
                    "but COMPLETE marker is missing."
                )

            else:
                message = (
                    f"Seed {seed:3d}: FAILED "
                    f"(rc={return_code}) — "
                    f"{elapsed / 60:.1f} min — "
                    f"finished {ts()}"
                )

            seed_log.write(f"\n[{ts()}] {message}\n")
            seed_log.flush()

            log(
                message,
                sweep_log,
            )

            return message

    except Exception as exc:
        elapsed = time.time() - start

        message = f"Seed {seed:3d}: CRASHED — {type(exc).__name__}: {exc} — {elapsed / 60:.1f} min"

        log(
            message,
            sweep_log,
        )

        return message


# ============================================================================
# Main
# ============================================================================


def main() -> int:

    os.makedirs(
        BASE_RESULTS,
        exist_ok=True,
    )

    summaries = []

    with open(
        SWEEP_LOG,
        "a",
        encoding="utf-8",
    ) as sweep_log:
        log(
            "\n" + "=" * 80,
            sweep_log,
        )

        log(
            f"SEED SWEEP STARTED {ts()}",
            sweep_log,
        )

        log(
            f"Seeds: {SEEDS}",
            sweep_log,
        )

        log(
            f"Seed timeout: {SEED_TIMEOUT_MINUTES:.1f} min",
            sweep_log,
        )

        log(
            f"Partition seed: {PARTITION_SEED}",
            sweep_log,
        )

        log(
            f"Partition alpha: {PARTITION_ALPHA}",
            sweep_log,
        )

        log(
            f"Num clients: {NUM_CLIENTS}",
            sweep_log,
        )

        log(
            f"NROWS: {NROWS}",
            sweep_log,
        )

        log(
            "=" * 80,
            sweep_log,
        )

        # --------------------------------------------------------------------
        # Create or validate sweep-level partition.
        # --------------------------------------------------------------------

        if not os.path.exists(PARTITION_FILE):
            create_partition(
                PARTITION_FILE,
                sweep_log,
            )

        else:
            partition_data = load_partition(
                PARTITION_FILE,
            )

            validate_partition(
                partition_data,
            )

            log(
                f"Found existing partition: {PARTITION_FILE}",
                sweep_log,
            )

            log(
                f"  Config: {partition_data['config']}",
                sweep_log,
            )

        log(
            "=" * 80,
            sweep_log,
        )

        # --------------------------------------------------------------------
        # Run every training seed using the SAME partition.
        # --------------------------------------------------------------------

        for seed in SEEDS:
            summary = run_seed(
                seed,
                sweep_log,
            )

            summaries.append(
                summary,
            )

        # --------------------------------------------------------------------
        # Final sweep summary.
        # --------------------------------------------------------------------

        log(
            "\n" + "=" * 80,
            sweep_log,
        )

        log(
            f"SEED SWEEP COMPLETE {ts()}",
            sweep_log,
        )

        log(
            "=" * 80,
            sweep_log,
        )

        for summary in summaries:
            log(
                summary,
                sweep_log,
            )

    # ------------------------------------------------------------------------
    # Exit nonzero if any seed is incomplete.
    # ------------------------------------------------------------------------

    incomplete = []

    for seed in SEEDS:
        results_dir = os.path.join(
            BASE_RESULTS,
            f"seed_{seed}",
        )

        if not seed_complete(results_dir):
            incomplete.append(seed)

    if incomplete:
        print(
            "\nSweep finished with incomplete seeds:",
            flush=True,
        )

        for seed in incomplete:
            print(
                f"  - seed {seed}",
                flush=True,
            )

        return 1

    print(
        "\nSUCCESS: All seeds completed.",
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
            "\nSeed sweep interrupted.",
            flush=True,
        )

        sys.exit(130)

    except Exception as exc:
        print(
            f"\nFATAL ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        sys.exit(1)
