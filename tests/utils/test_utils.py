"""
Unit tests for utils/utils.py
"""
import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path

from src.utils.utils import (
    set_random_seed,
    tensordatasets,
    dataloaders,
    decrease_lr,
    graphdataset,
    filter_args,
    get_optimal_params,
)


@pytest.mark.unit
class TestSetRandomSeed:
    """Tests for set_random_seed function"""

    def test_sets_numpy_seed(self):
        """Test that numpy random seed is set"""
        set_random_seed(42)
        val1 = np.random.rand()
        set_random_seed(42)
        val2 = np.random.rand()
        assert val1 == val2

    def test_sets_torch_seed(self):
        """Test that torch random seed is set"""
        set_random_seed(42)
        val1 = torch.rand(1).item()
        set_random_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2

    def test_different_seeds_different_values(self):
        """Test that different seeds produce different values"""
        set_random_seed(42)
        val1 = np.random.rand()
        set_random_seed(123)
        val2 = np.random.rand()
        assert val1 != val2


@pytest.mark.unit
class TestTensordatasets:
    """Tests for tensordatasets function"""

    @pytest.fixture
    def sample_train_df(self):
        """Create sample training dataframe"""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'label': [0, 1, 0, 1]
        })

    @pytest.fixture
    def sample_val_df(self):
        """Create sample validation dataframe"""
        return pd.DataFrame({
            'feature1': [1.5, 2.5],
            'feature2': [5.5, 6.5],
            'label': [0, 1]
        })

    @pytest.fixture
    def sample_test_df(self):
        """Create sample test dataframe"""
        return pd.DataFrame({
            'feature1': [3.5],
            'feature2': [7.5],
            'label': [1]
        })

    def test_train_only(self, sample_train_df):
        """Test with train data only"""
        trainset, valset, testset = tensordatasets(sample_train_df)

        assert trainset is not None
        assert valset is None
        assert testset is None
        assert len(trainset) == 4

    def test_train_and_val(self, sample_train_df, sample_val_df):
        """Test with train and validation data"""
        trainset, valset, testset = tensordatasets(
            sample_train_df, val_df=sample_val_df
        )

        assert trainset is not None
        assert valset is not None
        assert testset is None
        assert len(valset) == 2

    def test_all_splits(self, sample_train_df, sample_val_df, sample_test_df):
        """Test with all three splits"""
        trainset, valset, testset = tensordatasets(
            sample_train_df, val_df=sample_val_df, test_df=sample_test_df
        )

        assert trainset is not None
        assert valset is not None
        assert testset is not None
        assert len(testset) == 1

    def test_normalization(self, sample_train_df):
        """Test that normalization is applied"""
        trainset, _, _ = tensordatasets(sample_train_df, normalize=True)
        x, y = trainset[0]
        # Normalized values should be between 0 and 1
        assert x.min() >= 0
        assert x.max() <= 1

    def test_no_normalization(self, sample_train_df):
        """Test without normalization"""
        trainset, _, _ = tensordatasets(sample_train_df, normalize=False)
        x, y = trainset[0]
        # Original values preserved
        assert x[0].item() == 1.0

    def test_tensor_types(self, sample_train_df):
        """Test that tensors have correct types"""
        trainset, _, _ = tensordatasets(sample_train_df)
        x, y = trainset[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64


@pytest.mark.unit
class TestDataloaders:
    """Tests for dataloaders function"""

    @pytest.fixture
    def sample_datasets(self):
        """Create sample tensor datasets"""
        x_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))
        trainset = torch.utils.data.TensorDataset(x_train, y_train)

        x_val = torch.randn(20, 10)
        y_val = torch.randint(0, 2, (20,))
        valset = torch.utils.data.TensorDataset(x_val, y_val)

        x_test = torch.randn(30, 10)
        y_test = torch.randint(0, 2, (30,))
        testset = torch.utils.data.TensorDataset(x_test, y_test)

        return trainset, valset, testset

    def test_creates_all_loaders(self, sample_datasets):
        """Test that all dataloaders are created"""
        trainset, valset, testset = sample_datasets
        trainloader, valloader, testloader = dataloaders(trainset, valset, testset)

        assert trainloader is not None
        assert valloader is not None
        assert testloader is not None

    def test_batch_size(self, sample_datasets):
        """Test that batch size is respected"""
        trainset, valset, testset = sample_datasets
        trainloader, _, _ = dataloaders(trainset, valset, testset, batch_size=16)

        batch = next(iter(trainloader))
        assert batch[0].shape[0] == 16

    def test_none_datasets(self):
        """Test with None datasets"""
        trainloader, valloader, testloader = dataloaders(None, None, None)

        assert trainloader is None
        assert valloader is None
        assert testloader is None

    def test_with_sampler(self, sample_datasets):
        """Test with custom sampler (no shuffle)"""
        trainset, valset, testset = sample_datasets
        sampler = torch.utils.data.SequentialSampler(trainset)
        trainloader, _, _ = dataloaders(trainset, valset, testset, sampler=sampler)

        # Should work without error
        batch = next(iter(trainloader))
        assert batch[0] is not None


@pytest.mark.unit
class TestDecreaseLr:
    """Tests for decrease_lr function"""

    def test_decreases_learning_rate(self):
        """Test that learning rate is decreased"""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        decrease_lr(optimizer, factor=0.1)

        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.001)

    def test_custom_factor(self):
        """Test with custom factor"""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        decrease_lr(optimizer, factor=0.5)

        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.05)

    def test_multiple_param_groups(self):
        """Test with multiple parameter groups"""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam([
            {'params': model.weight, 'lr': 0.01},
            {'params': model.bias, 'lr': 0.001}
        ])

        decrease_lr(optimizer, factor=0.1)

        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.001)
        assert optimizer.param_groups[1]['lr'] == pytest.approx(0.0001)


@pytest.mark.unit
class TestGraphdataset:
    """Tests for graphdataset function"""

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data"""
        train_nodes = pd.DataFrame({
            'node': ['A', 'B', 'C', 'D'],
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'label': [0, 1, 0, 1]
        })
        train_edges = pd.DataFrame({
            'src': ['A', 'B', 'C'],
            'dst': ['B', 'C', 'D']
        })
        return train_nodes, train_edges

    def test_creates_train_graph(self, sample_graph_data):
        """Test that train graph is created"""
        train_nodes, train_edges = sample_graph_data
        trainset, valset, testset = graphdataset(
            train_nodes, train_edges,
            None, None, None, None
        )

        assert trainset is not None
        assert valset is None
        assert testset is None
        assert trainset.x.shape[0] == 4  # 4 nodes
        assert trainset.x.shape[1] == 2  # 2 features

    def test_edge_index_created(self, sample_graph_data):
        """Test that edge index is created correctly"""
        train_nodes, train_edges = sample_graph_data
        trainset, _, _ = graphdataset(
            train_nodes, train_edges,
            None, None, None, None,
            directed=True
        )

        # 3 edges in directed graph
        assert trainset.edge_index.shape[1] == 3

    def test_undirected_doubles_edges(self, sample_graph_data):
        """Test that undirected mode doubles edges"""
        train_nodes, train_edges = sample_graph_data
        trainset, _, _ = graphdataset(
            train_nodes, train_edges,
            None, None, None, None,
            directed=False
        )

        # 3 edges * 2 = 6 edges in undirected graph
        assert trainset.edge_index.shape[1] == 6

    def test_with_val_and_test(self, sample_graph_data):
        """Test with validation and test sets"""
        train_nodes, train_edges = sample_graph_data

        val_nodes = pd.DataFrame({
            'node': ['E', 'F'],
            'feature1': [5.0, 6.0],
            'feature2': [9.0, 10.0],
            'label': [1, 0]
        })
        val_edges = pd.DataFrame({
            'src': ['E'],
            'dst': ['F']
        })

        test_nodes = pd.DataFrame({
            'node': ['G'],
            'feature1': [7.0],
            'feature2': [11.0],
            'label': [1]
        })
        test_edges = pd.DataFrame({
            'src': [],
            'dst': []
        })

        trainset, valset, testset = graphdataset(
            train_nodes, train_edges,
            val_nodes, val_edges,
            test_nodes, test_edges
        )

        assert trainset is not None
        assert valset is not None
        assert testset is not None
        assert valset.x.shape[0] == 2
        assert testset.x.shape[0] == 1


@pytest.mark.unit
class TestFilterArgs:
    """Tests for filter_args function"""

    def test_filters_valid_args(self):
        """Test that only valid args are kept"""
        args = {
            'lr': 0.01,
            'betas': (0.9, 0.999),
            'invalid_param': 'should_be_removed'
        }

        filtered = filter_args(torch.optim.Adam, args)

        assert 'lr' in filtered
        assert 'betas' in filtered
        assert 'invalid_param' not in filtered

    def test_empty_dict_with_no_valid_args(self):
        """Test that empty dict returned when no valid args"""
        args = {'invalid1': 1, 'invalid2': 2}

        filtered = filter_args(torch.optim.Adam, args)

        # Only invalid args, so filtered should be empty (except maybe default params)
        assert 'invalid1' not in filtered
        assert 'invalid2' not in filtered


@pytest.mark.unit
class TestGetOptimalParams:
    """Tests for get_optimal_params function"""

    def test_reads_params_from_file(self, tmp_path):
        """Test reading params from file"""
        # Create a best_trials.txt file
        content = """learning_rate: 0.001
hidden_dim: 64
n_layers: 3
values: [0.95, 0.87]
"""
        (tmp_path / "best_trials.txt").write_text(content)

        config = {
            'learning_rate': 0.01,
            'hidden_dim': 32,
            'batch_size': 64  # Not in file
        }

        result = get_optimal_params(config, str(tmp_path))

        assert result['learning_rate'] == 0.001
        assert result['hidden_dim'] == 64
        assert result['batch_size'] == 64  # Unchanged
        assert 'n_layers' not in result  # Not in original config

    def test_handles_int_and_float(self, tmp_path):
        """Test that int and float types are correctly parsed"""
        content = """int_param: 42
float_param: 3.14
"""
        (tmp_path / "best_trials.txt").write_text(content)

        config = {'int_param': 0, 'float_param': 0.0}
        result = get_optimal_params(config, str(tmp_path))

        assert result['int_param'] == 42
        assert isinstance(result['int_param'], int)
        assert result['float_param'] == 3.14
        assert isinstance(result['float_param'], float)
