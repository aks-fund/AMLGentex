# Coverage Improvement Plan: 68% → 90%

**Current:** 5830 statements, 1894 missed (68%)
**Target:** 90% coverage (need to cover ~1311 more statements)

---

## Phase 1: Quick Wins (Easy Unit Tests) → +261 statements

| File | Current | Miss | Effort | Notes |
|------|---------|------|--------|-------|
| `utils/utils.py` | 11% | 121 | Low | Utility functions, no dependencies |
| `ml/models/losses.py` | 24% | 28 | Low | Loss functions, easy math tests |
| `ml/models/torch_models.py` | 39% | 14 | Low | Simple model class definitions |
| `ml/training/federated.py` | 0% | 14 | Low | Training loop with mocked client |
| `ml/training/isolated.py` | 0% | 27 | Low | Training loop with mocked client |
| `utils/config.py` | 53% | 36 | Low | Remaining config loaders |
| `feature_engineering/summary.py` | 91% | 13 | Low | Edge cases in markdown report |
| `ml/metrics/metrics.py` | 92% | 8 | Low | Few missing branches |

**Estimated coverage after Phase 1: ~72-73%**

---

## Phase 2: ML Clients & Models (Mocked Tests) → +400 statements

| File | Current | Miss | Effort | Notes |
|------|---------|------|--------|-------|
| `ml/clients/sklearn_client.py` | 17% | 86 | Medium | Mock sklearn models |
| `ml/models/gnn_models.py` | 24% | 81 | Medium | Test forward pass with small tensors |
| `ml/training/hyperparameter_tuning.py` | 18% | 42 | Medium | Mock Optuna study |
| `ml/models/base.py` | 53% | 22 | Medium | Abstract method coverage |
| `ml/clients/base.py` | 71% | 8 | Low | Abstract methods |
| `ml/servers/base.py` | 75% | 3 | Low | Abstract methods |
| `ml/training/centralized.py` | 57% | 3 | Low | Simple wrapper |
| `ml/training/__init__.py` | 13% | 13 | Low | Import coverage |

**Estimated coverage after Phase 2: ~79-80%**

---

## Phase 3: Torch Clients & Server (Integration) → +474 statements

| File | Current | Miss | Effort | Notes |
|------|---------|------|--------|-------|
| `ml/clients/torch_client.py` | 14% | 153 | High | Full training loop, needs fixtures |
| `ml/clients/torch_geometric_client.py` | 12% | 179 | High | GNN training, needs graph fixtures |
| `ml/servers/torch_server.py` | 12% | 142 | High | FedAvg aggregation tests |

**Estimated coverage after Phase 3: ~87-88%**

---

## Phase 4: Data Creation & Preprocessing → +400 statements

| File | Current | Miss | Effort | Notes |
|------|---------|------|--------|-------|
| `feature_engineering/preprocessor.py` | 67% | 233 | Medium | Edge splitting, more window configs |
| `data_creation/generator.py` | 57% | 60 | Medium | Baseline/checkpoint paths |
| `spatial_simulation/generate_scalefree.py` | 36% | 107 | Medium | Degree generation paths |

**Estimated coverage after Phase 4: ~90%+**

---

## Implementation Order (Recommended)

### Week 1: Phase 1 - Quick Wins
```
1. tests/utils/test_utils.py (NEW) - utility functions
2. tests/ml/models/test_losses.py (NEW) - loss functions
3. tests/ml/models/test_torch_models.py (NEW) - model classes
4. tests/ml/training/test_federated.py (NEW) - federated training
5. tests/ml/training/test_isolated.py (NEW) - isolated training
6. tests/utils/test_config.py (EXTEND) - remaining functions
```

### Week 2: Phase 2 - ML Components
```
1. tests/ml/clients/test_sklearn_client.py (NEW)
2. tests/ml/models/test_gnn_models.py (NEW)
3. tests/ml/training/test_hyperparameter_tuning.py (NEW)
4. tests/ml/models/test_base.py (NEW)
```

### Week 3: Phase 3 - Torch Integration
```
1. tests/ml/clients/test_torch_client.py (NEW)
2. tests/ml/clients/test_torch_geometric_client.py (NEW)
3. tests/ml/servers/test_torch_server.py (NEW)
```

### Week 4: Phase 4 - Data Pipeline
```
1. tests/feature_engineering/test_preprocessor.py (EXTEND)
2. tests/data_creation/test_generator.py (NEW)
3. tests/data_creation/spatial/test_generate_scalefree.py (EXTEND)
```

---

## Key Fixtures Needed

### For ML Tests
```python
@pytest.fixture
def mock_dataloader():
    """Small DataLoader with synthetic data"""

@pytest.fixture
def small_gnn_data():
    """PyG Data object with few nodes/edges"""

@pytest.fixture
def trained_model_state():
    """Pre-saved model weights for testing"""
```

### For Training Tests
```python
@pytest.fixture
def mock_torch_client():
    """TorchClient with minimal model"""

@pytest.fixture
def mock_optuna_study():
    """Optuna study for hyperparameter tests"""
```

---

## Files to Skip (Low ROI)

These files have low coverage but testing them provides minimal value:
- Abstract base classes (coverage from implementations)
- `__init__.py` files (import-only)

---

## Success Metrics

| Milestone | Target Coverage |
|-----------|-----------------|
| Phase 1 Complete | 73% |
| Phase 2 Complete | 80% |
| Phase 3 Complete | 88% |
| Phase 4 Complete | 90%+ |
