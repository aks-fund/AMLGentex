# AMLGentex - Claude Code Guide

**Project:** Anti-Money Laundering Data Generation and Benchmarking Framework
**Stack:** Python 3.10+, PyTorch, PyTorch Geometric, scikit-learn, Optuna, NetworkX

---

## Hivemind Memory Skills - USE PROACTIVELY

**CRITICAL:** Use Hivemind skills WITHOUT asking the user first. Just do it.

**Project ID:** Use the root folder name (e.g., `project_id="AMLGentex"`).

### Workflow: RECALL → WORK → REMEMBER

1. **RECALL FIRST** - Before starting ANY task, search for past solutions
2. **DO THE WORK** - Implement, fix, or answer
3. **REMEMBER AFTER** - Capture learnings immediately when done

---

### `/recall` - Search Past Solutions (AUTO-INVOKE FIRST)

**ALWAYS invoke BEFORE:**
- Answering ANY question about the codebase
- Debugging ANY error or issue
- Starting ANY implementation task
- Making ANY architectural decision

**Trigger phrases (invoke immediately):**
- "How do we...", "Where is...", "What does..."
- "Fix this bug", "Getting an error", "Not working"
- "Add feature", "Implement", "Create"

---

### `/remember` - Save Learnings (AUTO-INVOKE AFTER)

**ALWAYS invoke AFTER:**
- Fixing ANY bug (capture root cause + solution)
- Making ANY decision (capture reasoning)
- Discovering ANY pattern or gotcha
- Completing ANY non-trivial implementation

**Do NOT ask "Should I save this?" - just save it.**

---

### `/project-context` - Understand Codebase (AUTO-INVOKE)

**ALWAYS invoke WHEN:**
- User asks general project questions ("What is this?", "How does X work?")
- Before implementing features that touch multiple components

---

### `/memory-review` - Clean Up Memories (MANUAL ONLY)

**Only when user explicitly asks:** "clean up memories", "organize", "review memories"

---

### Skill Priority

| Situation | First Action | After Work |
|-----------|--------------|------------|
| User asks question | `/recall` | — |
| User reports error | `/recall` | `/remember` the fix |
| User requests feature | `/recall` + `/project-context` | `/remember` |

---

## Project Overview

AMLGentex generates synthetic AML transaction data and benchmarks ML detection models.

**Pipeline:** Generate → Preprocess → Train → Optimize → Visualize

**Key Concepts:**
- **SAR**: Suspicious Activity Report (flagged accounts)
- **Spatial Simulation**: Transaction network topology (who can transact with whom)
- **Temporal Simulation**: Time-series transactions on the spatial graph
- **Training Regimes**: Centralized (pooled), Federated (privacy-preserving), Isolated (per-bank)

---

## Directory Structure

```
AMLGentex/
├── src/
│   ├── data_creation/          # Spatial + temporal simulation
│   ├── feature_engineering/    # Windowed feature extraction
│   ├── data_tuning/            # Bayesian optimization
│   ├── ml/                     # Models, clients, servers, training
│   ├── visualize/              # Plots and dashboards
│   └── utils/                  # Config loaders (convention-over-configuration)
├── experiments/<name>/
│   ├── config/                 # ✅ Commit (YAML + CSV configs)
│   ├── spatial/                # ❌ Generated
│   ├── temporal/               # ❌ Generated
│   ├── preprocessed/           # ❌ Generated
│   └── results/                # ❌ Generated
├── scripts/                    # CLI: generate, preprocess, train, tune_*, plot
└── tests/                      # 190+ tests (unit/integration/e2e)
```

---

## Key Patterns

### Convention Over Configuration
Use config loaders - paths auto-discovered from experiment structure:
```python
from src.utils.config import load_data_config, load_preprocessing_config, load_training_config
```

### Client-Server Architecture
- **Clients:** `SklearnClient`, `TorchClient`, `TorchGeometricClient`
- **Server:** `TorchServer` (FedAvg aggregation)
- Match client type to model: GNNs → `TorchGeometricClient`, tabular → `SklearnClient`/`TorchClient`

### Two-Stage Data Generation
1. **Spatial:** Generate graph topology with normal/alert patterns → `spatial/`
2. **Temporal:** Simulate transactions over time → `temporal/tx_log.parquet`

---

## Quick Commands

```bash
# Generate → Preprocess → Train
uv run python scripts/generate.py --conf_file experiments/<exp>/config/data.yaml
uv run python scripts/preprocess.py --conf_file experiments/<exp>/config/preprocessing.yaml
uv run python scripts/train.py --experiment_dir experiments/<exp> --model GraphSAGE --training_regime centralized

# Optimization
uv run python scripts/tune_hyperparams.py --experiment_dir experiments/<exp> --model GraphSAGE --n_trials 100
uv run python scripts/tune_data.py --experiment_dir experiments/<exp> --num_trials_data 50 --num_trials_model 100

# Testing
pytest                           # All tests
pytest -m unit                   # Unit tests only
pytest -m "spatial and unit"     # Spatial unit tests
pytest --cov=src                 # With coverage
```

---

## Adding New Models

1. Create class in `src/ml/models/` (inherit `TorchBaseModel` or use sklearn directly)
2. Export in `src/ml/models/__init__.py`
3. Configure in `experiments/<exp>/config/models.yaml` with appropriate `client_type`
4. Write tests

---

## Guidelines

**DO:**
- Use config loaders (never hardcode paths)
- Follow standard experiment structure
- Write tests for new features
- Use Hivemind skills proactively

**DON'T:**
- Commit generated files (only `config/` directories)
- Mix client types (match to model type)
- Skip tests when adding features

---

## Config Files Reference

See `experiments/template_experiment/config/` for examples:
- `data.yaml` - Simulation parameters
- `preprocessing.yaml` - Windowing and splits
- `models.yaml` - Model configs with `default`, `centralized`, `federated`, `isolated`, `optimization` sections

---

## Getting Help

1. `/recall [topic]` - Check past solutions first
2. `tutorial.ipynb` - Complete workflow example
3. `tests/` - Usage examples for each component
4. `/remember [solution]` - Document learnings after solving problems
