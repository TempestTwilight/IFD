"""Configuration schema using dataclasses."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Dataset and partitioning configuration."""

    data_dir: str = "./data/raw"
    nrows: int | None = None
    synthetic_fallback: bool = False
    clip_value: float = 10.0

    # Partitioning
    num_clients: int = 10
    partition_method: str = "random"  # "random" or "dirichlet"
    partition_alpha: float = 0.5  # for dirichlet
    min_samples_per_client: int = 10
    min_class_samples_per_client: int = 2


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 512
    epochs_per_round: int = 10
    lr: float = 1e-3

    # Model
    hidden_dim: int = 128
    dropout: list[float] = field(default_factory=lambda: [0.2, 0.1])


@dataclass
class FLConfig:
    """Federated learning orchestration."""

    num_rounds: int = 20
    clients_per_round: int | None = None  # None = all clients


@dataclass
class AttackConfig:
    """Attack configuration."""

    num_adversaries: int = 0
    attack_type: str | None = None  # "sign_flip", "label_flip", "model_replace"

    # Attack-specific params
    oracle_eps: float = 0.3
    oracle_num_steps: int = 20
    grinding_drift_angle: float = 5.0
    grinding_warmup: int = 10
    spectral_gamma: float = 0.95


@dataclass
class DefenseConfig:
    """Defense layer configuration."""

    disable_layer1: bool = False
    disable_layer2: bool = False
    disable_layer3: bool = False

    # Layer 1
    z_thresh_l1: float = 3.3

    # Layer 2
    gamma_l2: float = 0.95

    # Layer 3
    slope_l3: float = 6.0
    h_local_l3: float = 1.2
    h_global_l3: float = 1.2
    mu0_local_l3: float = 0.33
    mu0_global_l3: float = 0.28
    warmup_rounds_l3: int = 5


@dataclass
class BaselineConfig:
    """Baseline strategy configuration."""

    baseline: str | None = None  # None = IFD cascade, or b1-b9

    # Baseline-specific
    krum_f: int = 1
    trimmed_beta: float = 0.1
    foolsgold_temp: float = 5.0
    dpfl_clip_norm: float = 1.0
    dpfl_noise_std: float = 0.1
    fldetector_threshold: float = 0.5


@dataclass
class OutputConfig:
    """Output and logging configuration."""

    results_dir: str = "./results"
    save_model: str | None = None
    run_label: str | None = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    seed: int = 44

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict

        return asdict(self)
