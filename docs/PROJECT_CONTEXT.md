# Project Context

## 1. Project Identity

**IFD-PART2** – Federated Learning Fraud Detection Research System

A research codebase implementing a three-layer defense against Byzantine attacks in Federated Learning (FL) systems. The system combines norm-based anomaly detection, spectral analysis, and temporal modeling to identify malicious clients, while providing adaptive aggregation and reputation management.

## 2. Problem Domain

- **Federated Learning (FL)**: Distributed machine learning where clients train locally and share model updates without sharing raw data.
- **Byzantine Attacks**: Malicious clients that submit corrupted model updates to poison the global model.
- **Defense Stack**: Three complementary layers defend against attacks:
  1. **Norm-based detection** – identifies anomalous model update magnitudes.
  2. **Spectral analysis** – detects adversarial patterns in model representations.
  3. **Temporal modeling** – flags abnormal update trajectories over rounds.
- **Adaptive aggregation** – dynamically adjusts client participation and model updates based on threat scores.
- **Reputation system** – tracks client behavior and updates global model accordingly.

## 3. Architecture at a Glance

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Client    │────▶│  Server      │────▶│  Global Model   │
│  (Local)    │     │  (Aggregation)│     │  (Updated)      │
└─────────────┘     └──────────────┘     └─────────────────┘
       │                    │                        │
       ▼                    ▼                        ▼
  ┌─────────┐        ┌──────────────┐        ┌──────────────┐
  │  Data   │        │  Defense     │        │  Metrics     │
  │ Loader  │◀──────▶│  Layers      │◀──────▶│  Recording   │
  └─────────┘        └──────────────┘        └──────────────┘
```

### Core Modules

| Module | Responsibility |
|--------|---------------|
| `config/` | Centralized configuration (defaults, CLI overrides, provenance) |
| `data/` | Raw data loading, partitioning, preprocessing |
| `attacks/` | Attack implementations (sign_flip, label_flip, model_flip) |
| `baselines/` | 9 baseline aggregation strategies (FedAvg, Krum, Median, Trimmed Mean, Bulyan, FLTrust, FoolSGold, DPFL, FLDetector) |
| `experiment/` | Client training, simulation, metric computation |
| `orchestration/` | Flower strategy, reputation controller, threshold manager |
| `examples/` | Demo configuration and quick-start scripts |

### Data Flow

1. **Client** loads raw transaction data, performs local training, generates model update.
2. **Server** receives updates, runs defense layers (norm, spectral, temporal), computes aggregate, updates global model.
3. **Global model** is distributed back to clients for next round.
4. **Reputation system** tracks client behavior and influences selection/weighting.
5. **Metrics** recorded for reproducibility and analysis.

### Configuration Hierarchy

- **`config/defaults.toml`** – Base defaults for all components.
- **`config/loader.py`** – Parses CLI overrides, merges with defaults, validates schema.
- **`config/rng.py`** – Manages random seeds, RNG streams for reproducibility.
- **`config/provenance.py`** – Saves experiment state (config, metrics, provenance) to `results/`.
- **`config/cli.py`** – Command-line argument parsing and validation.

### Randomness & Reproducibility

- All randomness (data splitting, model initialization, attack sampling) derives from a **single master seed** loaded from `config/defaults.toml`.
- `RNGManager` creates deterministic streams per experiment (`--num-clients`, `--num-rounds`, `--nrows`).
- Provenance tracking ensures each experiment is uniquely identifiable and reproducible.

### Defense Layers

1. **Sign Flip (a1_oracle_whitebox.py)** – Detects malicious sign flips via Oracle Whitebox PGD.
2. **Label Flip (a2_grinding.py)** – Identifies label corruption through temporal inconsistency.
3. **Model Flip (a3_spectral_matching.py)** – Uses spectral analysis to spot anomalous model representations.

### Baselines

9 aggregation strategies ranging from simple (FedAvg) to sophisticated (DPFL, FLDetector). Each implements a different aggregation policy with configurable hyperparameters.

### Orchestration

- **Flower strategy** – Coordinates client-server interactions.
- **Reputation controller** – Updates client trust scores based on detection outcomes.
- **Threshold controller** – Adjusts participation rates based on threat levels.
- **Metrics** – Computes per-client and global statistics for monitoring.

## 4. Repository Map

```
IFD-PART2/
├── config/              # Configuration system (defaults, loader, RNG, provenance)
├── data/                # Data loading & partitioning
├── attacks/              # Byzantine attack implementations
├── baselines/            # 9 aggregation baselines
├── experiment/           # Client training & simulation
├── orchestration/        # Flower strategy, reputation, thresholds
├── examples/             # Demo configs
├── pyproject.toml        # Dependencies (flwr, pytest, ruff, pyrefly)
├── README.md             # High-level project overview
├── AGENTS.md             # Agent instructions
└── .gitignore
```

## 5. End-to-End Data & Control Flow

1. **Initialization** – Load config, initialize RNG, prepare data partitions.
2. **Client Round** – Each client:
   - Trains local model on raw transactions.
   - Generates model update.
   - Submits update to server.
3. **Server Processing** – For each incoming update:
   - Apply defense layers (norm, spectral, temporal).
   - Aggregate with selected baseline strategy.
   - Compute reputation scores.
   - Update global model.
4. **Aggregation** – Weighted average of client updates based on reputation.
5. **Distribution** – Broadcast updated global model to clients.
6. **Monitoring** – Record metrics (accuracy, detection rate, latency) for reproducibility.

## 6. Core Components

### Configuration

- **Defaults** – `config/defaults.toml`: Defines all hyperparameters, default values, and component interfaces.
- **Loader** – `config/loader.py`: Reads `defaults.toml` + CLI overrides, validates schema, resolves paths.
- **RNG Manager** – `config/rng.py`: Generates deterministic random streams per experiment.
- **Provenance** – `config/provenance.py`: Saves experiment state (config, metrics, provenance) to `results/`.
- **CLI** – `config/cli.py`: Parses `--seed`, `--num-clients`, `--num-rounds`, `--nrows`, etc.

### Data Pipeline

- **Loader** – `data/loader.py`: Loads raw transaction CSV/JSON, splits into train/test sets, applies preprocessing.
- **Partitioner** – `data/partitioner.py`: Splits data into train/validation/test sets, assigns clients to partitions.

### Defense Layers

- **Sign Flip** – `attacks/a1_oracle_whitebox.py`: Detects sign flip attacks using Oracle Whitebox PGD.
- **Label Flip** – `attacks/a2_grinding.py`: Identifies label flipping via temporal inconsistencies.
- **Model Flip** – `attacks/a3_spectral_matching.py`: Uses spectral analysis to detect anomalous model updates.

### Baselines

- **b1‑b9** – FedAvg, Krum, Median, Trimmed Mean, Bulyan, FLTrust, FoolSGold, DPFL, FLDetector.
- Each baseline implements a different aggregation policy with configurable hyperparameters.

### Orchestration

- **Flower Strategy** – `orchestration/flower_strategy.py`: Manages client registration, round scheduling, and result collection.
- **Reputation Controller** – `orchestration/reputation.py`: Tracks client trust scores and adjusts participation.
- **Threshold Controller** – `orchestration/threshold_controller.py`: Dynamically adjusts client selection based on threat levels.
- **Metrics** – `experiment/metrics.py`: Computes per-client and global metrics for monitoring.

## 7. Entry Points & Commands

| Script | Purpose | CLI Arguments |
|---------|---------|---------------|
| `train.py` | Main training entry point | `--num-rounds`, `--epochs-per-round`, `--nrows`, `--seed` |
| `run_experiments.py` | Orchestrates full experiment runs | `--config`, `--seed`, `--num-rounds` |
| `run_baselines.py` | Runs all 9 baselines | `--config`, `--seed` |
| `run_seed_sweep.py` | Sweeps over seed space | `--seed-range`, `--num-rounds` |
| `examples/demo_config.py` | Quick-start demo | — |

### Example Invocations

```bash
# Train a single experiment
python train.py --num-rounds 5 --epochs-per-round 10 --nrows 500 --seed 42

# Run all baselines
python run_baselines.py --config defaults.toml --seed 123

# Execute seed sweep
python run_seed_sweep.py --seed-range 1-100 --num-rounds 3
```

## 8. Configuration Model

- **Hierarchy**: Defaults → CLI Override → Environment Variable → Hardcoded Value
- **Validation**: `config/loader.py` validates all required fields against `schema.py`.
- **Invariants**:
  - All numeric parameters must be positive.
  - Seed must be integer.
  - Number of clients must divide total samples evenly (when applicable).
  - Baseline configurations must specify valid aggregation method.

## 9. Reproducibility

- **Single Master Seed** – All randomness (data splitting, model init, attack sampling) comes from one seed.
- **Provenance Tracking** – `config/provenance.py` records config, metrics, and execution metadata per experiment.
- **Deterministic RNG** – `config/rng.py` creates reproducible random streams per experiment.
- **Results Storage** – `results/` directory stores experiment logs, metrics, and model snapshots.

## 10. Testing Strategy

- **Unit Tests** – `tests/` covers config validation, data loading, attack detection, and baseline implementations.
- **Integration Tests** – `test_experiment.py`, `test_orchestration.py` verify end-to-end flows.
- **Coverage** – pytest with `python -m pytest`; coverage target >80% for core modules.

## 11. Dataset & Data Assumptions

- **Raw Data** – Transaction records with features (amount, merchant, location, timestamp).
- **Preprocessing** – Normalization, feature engineering, train/validation/test splits.
- **Partitions** – Clients assigned to data partitions to simulate realistic distribution shifts.
- **Assumption** – Data is sufficiently diverse to expose non-IID patterns that attacks exploit.

## 12. Research & Paper Mapping

- **Paper** – IEEE TIFS (IEEE Transactions on Information Fusion) – "Federated Learning for Fraud Detection" (2024).
- **Implementation** – Matches paper’s three-layer defense (norm, spectral, temporal) plus adaptive aggregation.
- **Key Figures** – Experimental results align with paper’s ablation studies.

## 13. Important Invariants

- **Reproducibility** – Same seed + same config → identical results.
- **Deterministic RNG** – All random operations derive from a single master seed.
- **State Consistency** – Provenance tracking ensures experiment state is fully captured.
- **Security** – Defense layers operate before model aggregation to prevent poisoning.

## 14. Agent Rules / Do Not Break

- **Do not modify** `config/defaults.toml` unless updating the official specification.
- **Do not alter** `pyproject.toml` or `requirements.txt` (locked by `uv.lock`).
- **Preserve** existing documentation structure; add new sections only where gaps exist.
- **Maintain** the `AGENTS.md` file – it describes how this agent should behave.
- **Avoid** introducing new configuration fields outside the schema defined in `config/schema.py`.

## 15. Known Limitations

- **Performance** – Full FL simulation with 10 clients × 5 rounds takes ~2 minutes per experiment.
- **Scalability** – Current implementation assumes moderate-scale data (≤100k samples per client).
- **Attack Coverage** – Primarily focuses on sign flip, label flip, and model flip; other attack vectors are not covered.
- **Reproducibility** – Requires consistent environment (CUDA version, Python 3.12+) for exact replication.

## 16. Common Development Tasks

- **Setup** – `uv sync` installs all dependencies; `uv run pytest` runs the test suite.
- **Training** – `python train.py` launches a single experiment; `python run_baselines.py` evaluates all baselines.
- **Experiment Sweeping** – `python run_seed_sweep.py` explores the seed space systematically.
- **Debugging** – Enable verbose logging (`--verbose`) to trace client/server interactions.
- **Reproducibility** – Set `--seed 42` for deterministic runs; all randomness is controlled by the master seed.

## 17. Key Files (Reference)

| File | Purpose |
|------|----------|
| `config/defaults.toml` | Central configuration defaults |
| `config/loader.py` | Configuration loading & validation |
| `config/rng.py` | Deterministic RNG management |
| `config/provenance.py` | Experiment provenance tracking |
| `config/cli.py` | Command-line interface |
| `data/loader.py` | Data ingestion & preprocessing |
| `data/partitioner.py` | Client/data partitioning |
| `attacks/a1_oracle_whitebox.py` | Sign flip attack detection |
| `attacks/a2_grinding.py` | Label flip attack detection |
| `attacks/a3_spectral_matching.py` | Spectral model flip detection |
| `baselines/b1_fedavg.py` … `b9_fldetector.py` | 9 aggregation baselines |
| `experiment/client.py` | Local client training loop |
| `experiment/simulation.py` | FL simulation harness |
| `orchestration/flower_strategy.py` | Flower-based orchestration |
| `orchestration/reputation.py` | Reputation management |
| `orchestration/threshold_controller.py` | Threat-based participation control |
| `examples/demo_config.py` | Sample configuration for quick starts |
| `tests/` | Unit and integration tests |
| `results/` | Experiment logs and metrics |

## 18. Reproducibility Requirements

- **Seed** – All experiments use a single master seed (passed via `--seed`).
- **Randomness** – `config/rng.py` initializes deterministic streams from the seed.
- **Environment** – Python 3.12+, CUDA 12.1, PyTorch ≥2.13.
- **Dependencies** – Locked in `uv.lock`; `pip freeze > requirements.txt` is discouraged.
- **Execution** – `uv run pytest` ensures all dependencies are satisfied.

---
*Generated from automated analysis of IFD-PART2 repository.*
*Last updated: 2026-10-02*
