#!/usr/bin/env python3
"""
Baseline Experiment Orchestrator — IFD-PART2
=============================================

Runs Clean + Attack Robustness suite for B1 (FedAvg) and B2 (Krum),
saving results to:

    results/baselines/<baseline_name>/<experiment_label>.json

10 experiments per baseline = 20 total.

Skip logic: if the output JSON already exists AND contains non-empty
metrics_distributed (i.e. was completed with full metric logging),
the experiment is skipped. This lets you safely re-run after a crash.

Usage:
    python run_baselines.py

Environment variables:
    RESULTS_DIR   default: ./results
    NUM_CLIENTS   default: 10
    NUM_ROUNDS    default: 50
    NROWS         default: 150000
"""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from config import load_config, save_provenance

# ============================================================================
# Configuration
# ============================================================================

# Base config from environment variables
BASE_CONFIG = load_config(
    overrides={
        "data": {
            "num_clients": int(os.environ.get("NUM_CLIENTS", "10")),
            "nrows": int(os.environ.get("NROWS", "150000")),
        },
        "fl": {
            "num_rounds": int(os.environ.get("NUM_ROUNDS", "50")),
        },
        "training": {
            "batch_size": 512,
            "epochs_per_round": 10,
        },
        "output": {
            "results_dir": os.environ.get("RESULTS_DIR", "./results"),
        },
    }
)

RATIOS = [0.10, 0.20, 0.40]

ATTACKS = ["sign_flip", "label_flip", "model_replace"]

ATTACK_LABELS = {
    "sign_flip": "SignFlip",
    "label_flip": "LabelFlip",
    "model_replace": "ModelReplace",
}

# Only B1 and B2
BASELINES = ["b1_fedavg", "b2_krum"]


# ============================================================================
# Paths
# ============================================================================

os.makedirs(BASE_CONFIG.output.results_dir, exist_ok=True)

LOG_PATH = os.path.join(BASE_CONFIG.output.results_dir, "baselines_experiment_log.txt")

BASE_CMD = [
    sys.executable,
    "train.py",
    "--num-clients",
    str(BASE_CONFIG.data.num_clients),
    "--num-rounds",
    str(BASE_CONFIG.fl.num_rounds),
    "--batch-size",
    str(BASE_CONFIG.training.batch_size),
    "--epochs-per-round",
    str(BASE_CONFIG.training.epochs_per_round),
    "--nrows",
    str(BASE_CONFIG.data.nrows),
]


# ============================================================================
# Utility
# ============================================================================


def timestamp():
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_log(message):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp()}] {message}\n")
    print(f"[{timestamp()}] {message}")


def load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def experiment_already_completed(out_path):
    data = load_json(out_path)
    if not data:
        return False
    metrics_dist = data.get("metrics_distributed", {})
    return bool(metrics_dist)


# ============================================================================
# Experiment Builders
# ============================================================================


def build_clean_config(baseline_name: str):
    """Build config override dict for clean baseline experiment."""
    return {
        "data": {
            "num_clients": BASE_CONFIG.data.num_clients,
            "nrows": BASE_CONFIG.data.nrows,
        },
        "training": {
            "batch_size": BASE_CONFIG.training.batch_size,
            "epochs_per_round": BASE_CONFIG.training.epochs_per_round,
        },
        "fl": {
            "num_rounds": BASE_CONFIG.fl.num_rounds,
        },
        "attack": {
            "num_adversaries": 0,
            "attack_type": None,
        },
        "baseline": {
            "baseline": baseline_name,
        },
        "output": {
            "results_dir": BASE_CONFIG.output.results_dir,
        },
    }


def build_attack_config(baseline_name: str, attack_type: str, ratio: float):
    """Build config override dict for attack baseline experiment."""
    num_adv = max(1, int(BASE_CONFIG.data.num_clients * ratio))
    return {
        "data": {
            "num_clients": BASE_CONFIG.data.num_clients,
            "nrows": BASE_CONFIG.data.nrows,
        },
        "training": {
            "batch_size": BASE_CONFIG.training.batch_size,
            "epochs_per_round": BASE_CONFIG.training.epochs_per_round,
        },
        "fl": {
            "num_rounds": BASE_CONFIG.fl.num_rounds,
        },
        "attack": {
            "num_adversaries": num_adv,
            "attack_type": attack_type,
        },
        "baseline": {
            "baseline": baseline_name,
        },
        "output": {
            "results_dir": BASE_CONFIG.output.results_dir,
        },
    }


# ============================================================================
# Experiment Runner
# ============================================================================


def run_experiment(baseline_name, label, cmd, out_path):
    """
    Run a single baseline experiment.

    Args:
        baseline_name: e.g. "b1_fedavg"
        label: e.g. "Clean"
        cmd: command list to run
        out_path: path where train.py writes JSON output
    """
    write_log(f"▶ [{baseline_name}] {label}")

    if experiment_already_completed(out_path):
        write_log(f"  ✓ Already completed: {out_path}")
        return

    if os.path.isfile(out_path):
        os.remove(out_path)

    # Build config for provenance
    if "num-adversaries 0" in " ".join(cmd):
        config_overrides = build_clean_config(baseline_name)
    else:
        # Extract attack params from command
        attack_idx = cmd.index("--attack") + 1
        attack_type = cmd[attack_idx]
        adv_idx = cmd.index("--num-adversaries") + 1
        num_adv = int(cmd[adv_idx])
        ratio = num_adv / BASE_CONFIG.data.num_clients
        config_overrides = build_attack_config(baseline_name, attack_type, ratio)

    config = load_config(overrides=config_overrides)
    provenance_dir = Path(out_path).parent / f"{label.lower().replace(' ', '_')}_provenance"

    write_log(f"  Running: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            if os.path.isfile(out_path):
                data = load_json(out_path)
                if data and data.get("metrics_distributed"):
                    write_log(f"  ✓ Success in {elapsed:.1f}s")
                    # Save provenance
                    metrics = {
                        "baseline": baseline_name,
                        "experiment": label,
                        **data.get("metrics_distributed", {}),
                    }
                    save_provenance(config, str(provenance_dir), metrics=metrics)
                else:
                    write_log(f"  ✗ Completed but output incomplete: {out_path}")
            else:
                write_log(f"  ✗ Completed but no output file: {out_path}")
        else:
            write_log(f"  ✗ Failed (exit={result.returncode}) after {elapsed:.1f}s")
            write_log(f"    stderr: {result.stderr[-500:]}")

    except Exception as e:
        elapsed = time.time() - start
        write_log(f"  ✗ Exception after {elapsed:.1f}s: {e}")


# ============================================================================
# Main Orchestration
# ============================================================================


def main():
    write_log("=" * 80)
    write_log("Baseline Experiment Orchestrator")
    write_log("=" * 80)
    write_log(f"RESULTS_DIR: {BASE_CONFIG.output.results_dir}")
    write_log(f"NUM_CLIENTS: {BASE_CONFIG.data.num_clients}")
    write_log(f"NUM_ROUNDS: {BASE_CONFIG.fl.num_rounds}")
    write_log(f"NROWS: {BASE_CONFIG.data.nrows}")
    write_log("")

    for baseline_name in BASELINES:
        baseline_dir = os.path.join(BASE_CONFIG.output.results_dir, "baselines", baseline_name)
        os.makedirs(baseline_dir, exist_ok=True)

        write_log(f"\n{'=' * 80}")
        write_log(f"Baseline: {baseline_name.upper()}")
        write_log(f"{'=' * 80}")

        # 1) Clean
        out_clean = os.path.join(baseline_dir, "clean.json")
        cmd_clean = BASE_CMD + [
            "--baseline",
            baseline_name,
            "--num-adversaries",
            "0",
            "--output",
            out_clean,
        ]
        run_experiment(baseline_name, "Clean", cmd_clean, out_clean)

        # 2) 9 attack experiments
        for attack in ATTACKS:
            for ratio in RATIOS:
                num_adv = max(1, int(BASE_CONFIG.data.num_clients * ratio))
                attack_label = ATTACK_LABELS[attack]
                label = f"{attack_label}_{int(ratio * 100):02d}"
                out_attack = os.path.join(baseline_dir, f"{label}.json")

                cmd_attack = BASE_CMD + [
                    "--baseline",
                    baseline_name,
                    "--num-adversaries",
                    str(num_adv),
                    "--attack",
                    attack,
                    "--output",
                    out_attack,
                ]
                run_experiment(baseline_name, label, cmd_attack, out_attack)

    write_log("\n" + "=" * 80)
    write_log("All baseline experiments complete.")
    write_log("=" * 80)


if __name__ == "__main__":
    main()
