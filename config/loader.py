"""Configuration loader with TOML support and CLI overrides."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "Python <3.11 requires 'tomli' package. Install with: pip install tomli"
        ) from None

from config.schema import (
    AttackConfig,
    BaselineConfig,
    DataConfig,
    DefenseConfig,
    ExperimentConfig,
    FLConfig,
    OutputConfig,
    TrainingConfig,
)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> ExperimentConfig:
    """Load configuration from TOML file with optional overrides.

    Args:
        config_path: Path to TOML config file. If None, use defaults.
        overrides: Dictionary of overrides (e.g., from CLI args).

    Returns:
        Fully resolved ExperimentConfig.
    """
    # Start with defaults
    config_dict = {}

    # Load TOML if provided
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path, "rb") as f:
            config_dict = tomllib.load(f)

    # Apply overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    # Build config objects
    return ExperimentConfig(
        seed=config_dict.get("seed", 44),
        data=_build_data_config(config_dict.get("data", {})),
        training=_build_training_config(config_dict.get("training", {})),
        fl=_build_fl_config(config_dict.get("fl", {})),
        attack=_build_attack_config(config_dict.get("attack", {})),
        defense=_build_defense_config(config_dict.get("defense", {})),
        baseline=_build_baseline_config(config_dict.get("baseline", {})),
        output=_build_output_config(config_dict.get("output", {})),
    )


def _deep_merge(base: dict, updates: dict) -> dict:
    """Deep merge updates into base."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _build_data_config(d: dict) -> DataConfig:
    return DataConfig(
        data_dir=d.get("data_dir", "./data/raw"),
        nrows=d.get("nrows"),
        synthetic_fallback=d.get("synthetic_fallback", False),
        clip_value=d.get("clip_value", 10.0),
        num_clients=d.get("num_clients", 10),
        partition_method=d.get("partition_method", "random"),
        partition_alpha=d.get("partition_alpha", 0.5),
        min_samples_per_client=d.get("min_samples_per_client", 10),
        min_class_samples_per_client=d.get("min_class_samples_per_client", 2),
    )


def _build_training_config(d: dict) -> TrainingConfig:
    return TrainingConfig(
        batch_size=d.get("batch_size", 512),
        epochs_per_round=d.get("epochs_per_round", 10),
        lr=d.get("lr", 1e-3),
        hidden_dim=d.get("hidden_dim", 128),
        dropout=d.get("dropout", [0.2, 0.1]),
    )


def _build_fl_config(d: dict) -> FLConfig:
    return FLConfig(
        num_rounds=d.get("num_rounds", 20),
        clients_per_round=d.get("clients_per_round"),
    )


def _build_attack_config(d: dict) -> AttackConfig:
    return AttackConfig(
        num_adversaries=d.get("num_adversaries", 0),
        attack_type=d.get("attack_type"),
        oracle_eps=d.get("oracle_eps", 0.3),
        oracle_num_steps=d.get("oracle_num_steps", 20),
        grinding_drift_angle=d.get("grinding_drift_angle", 5.0),
        grinding_warmup=d.get("grinding_warmup", 10),
        spectral_gamma=d.get("spectral_gamma", 0.95),
    )


def _build_defense_config(d: dict) -> DefenseConfig:
    return DefenseConfig(
        disable_layer1=d.get("disable_layer1", False),
        disable_layer2=d.get("disable_layer2", False),
        disable_layer3=d.get("disable_layer3", False),
        z_thresh_l1=d.get("z_thresh_l1", 3.3),
        gamma_l2=d.get("gamma_l2", 0.95),
        slope_l3=d.get("slope_l3", 6.0),
        h_local_l3=d.get("h_local_l3", 1.2),
        h_global_l3=d.get("h_global_l3", 1.2),
        mu0_local_l3=d.get("mu0_local_l3", 0.33),
        mu0_global_l3=d.get("mu0_global_l3", 0.28),
        warmup_rounds_l3=d.get("warmup_rounds_l3", 5),
    )


def _build_baseline_config(d: dict) -> BaselineConfig:
    return BaselineConfig(
        baseline=d.get("baseline"),
        krum_f=d.get("krum_f", 1),
        trimmed_beta=d.get("trimmed_beta", 0.1),
        foolsgold_temp=d.get("foolsgold_temp", 5.0),
        dpfl_clip_norm=d.get("dpfl_clip_norm", 1.0),
        dpfl_noise_std=d.get("dpfl_noise_std", 0.1),
        fldetector_threshold=d.get("fldetector_threshold", 0.5),
    )


def _build_output_config(d: dict) -> OutputConfig:
    return OutputConfig(
        results_dir=d.get("results_dir", "./results"),
        save_model=d.get("save_model"),
        run_label=d.get("run_label"),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save resolved config to YAML for provenance."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON for simplicity (YAML would need PyYAML)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
