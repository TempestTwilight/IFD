# AGENTS.md
## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Project overview

IFD-PART2 — federated-learning fraud detection research codebase (IEEE TIFS paper). Three-layer defense against Byzantine attacks in FL, with 9 baseline comparisons.

## Toolchain

- **Python:** 3.12 (`.python-version`), `requires-python = ">=3.12,<3.14"`
- **Package manager:** `uv` (no pip). Deps in `pyproject.toml`, lockfile `uv.lock`.
- **Linters:** `ruff` + `pyrefly` (both in dev deps)
- **Tests:** `pytest` with `pytest-cov`
- **Docker:** CUDA 12.1 GPU image, `uv`-based (see Dockerfile)

## Commands

```bash
uv sync                          # install deps (run once)
uv run pytest                    # all tests with coverage
uv run pytest tests/test_foo.py  # single test file
uv run ruff check .              # lint (must pass clean)
uv run pyrefly check .           # type check (must pass 0 errors)
uv run python train.py --help    # main training entry point
```

Both `ruff check` and `pyrefly check` must pass with 0 errors before any PR.

## Repo layout

```
train.py                  # main FL training entry point
run_experiments.py        # 22-experiment orchestrator (calls train.py)
run_baselines.py          # baseline comparison orchestrator
run_seed_sweep.py         # reproducibility seed sweep
config/                   # config system (dataclasses + TOML, stdlib only)
  schema.py               #   ExperimentConfig and nested dataclasses
  loader.py               #   load_config() — TOML + CLI override merging
  rng.py                  #   RNGManager — SeedSequence-based RNG streams
  defaults.toml           #   single source of truth for all defaults
  provenance.py           #   save resolved config + git info per run
  cli.py                  #   argparse → override dict converter
attacks/                  # Byzantine attack implementations
  a1_oracle_whitebox.py   #   sign_flip  (OracleWhiteBoxPGD)
  a2_grinding.py          #   label_flip (TemporalGrinding)
  a3_spectral_matching.py #   model_flip (SpectralMatching)
baselines/                # 9 aggregation baselines (b1–b9)
layers/                   # 3-layer defense (norm/cosine, spectral, temporal)
experiment/               # FL simulation, client, metrics, ablation
orchestration/            # Flower strategy, reputation, thresholds
data/                     # loader + partitioners (Dirichlet, Random)
  raw/                    #   IEEE CIS data (gitignored, downloaded or synthetic)
tests/                    # pytest suite — pythonpath=["."]
results/                  # run outputs (gitignored)
```

## Config system

All experiment parameters live in `config/defaults.toml`. Entry points load config via:
```python
from config import load_config, RNGManager, save_provenance

config = load_config()  # defaults only
config = load_config(overrides={"seed": 99})  # with overrides
rng = RNGManager(config.seed)  # deterministic RNG streams
```
No Hydra, no Pydantic — stdlib `dataclasses` + `tomllib` only.

## Conventions

- `torch.nn.functional` imported as `torch_nn_functional` (ruff N812 alias rule)
- Attacks map: `a1` = sign_flip, `a2` = label_flip, `a3` = model_flip
- Baselines: `b1`–`b9` (FedAvg, Krum, Median, TrimmedMean, Bulyan, FLTrust, FoolsGold, DPFL, FLDetector)
- ruff config: `line-length = 100`, select `E,F,I,UP,B,SIM,N,RUF`
- `pytest` uses `pythonpath = ["."]` — imports are from repo root

## Pitfalls

- `data/raw/` is gitignored — loader falls back to synthetic data when CSVs are missing
- `results/` is also gitignored — don't expect prior run outputs in a fresh clone
- `.docs/` is gitignored — internal planning docs, not part of the project
- ruff forbids `import torch.nn.functional as F` — use `torch_nn_functional`
- pyrefly warns on `Optional` gradients — guard with `if grad is not None:` before use
- The orchestrators (`run_experiments.py`, `run_baselines.py`, `run_seed_sweep.py`) call `train.py` as subprocesses — changes to train.py's CLI args affect all three
