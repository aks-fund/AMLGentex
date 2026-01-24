"""
Unit tests for ml/models/torch_models.py
"""
import pytest
import torch

from src.ml.models.torch_models import LogisticRegressor, MLP


@pytest.mark.unit
class TestLogisticRegressor:
    """Tests for LogisticRegressor model"""

    def test_init(self):
        """Test initialization"""
        model = LogisticRegressor(input_dim=10, output_dim=2)

        assert model.linear.in_features == 10
        assert model.linear.out_features == 2

    def test_forward(self):
        """Test forward pass"""
        model = LogisticRegressor(input_dim=10, output_dim=2)
        x = torch.randn(32, 10)

        output = model(x)

        assert output.shape == (32, 2)

    def test_forward_single_sample(self):
        """Test forward pass with single sample"""
        model = LogisticRegressor(input_dim=5, output_dim=3)
        x = torch.randn(1, 5)

        output = model(x)

        # squeeze() removes the batch dimension for single samples
        assert output.shape == (3,)

    def test_binary_output(self):
        """Test with binary classification output"""
        model = LogisticRegressor(input_dim=10, output_dim=1)
        x = torch.randn(16, 10)

        output = model(x)

        # With output_dim=1, squeeze removes the dimension
        assert output.shape == (16,)

    def test_gradient_flows(self):
        """Test that gradients flow through the model"""
        model = LogisticRegressor(input_dim=10, output_dim=2)
        x = torch.randn(8, 10)
        target = torch.randint(0, 2, (8,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        assert model.linear.weight.grad is not None


@pytest.mark.unit
class TestMLP:
    """Tests for MLP model"""

    def test_init(self):
        """Test initialization"""
        model = MLP(input_dim=10, n_hidden_layers=2, hidden_dim=32, output_dim=2)

        assert model.input_layer.in_features == 10
        assert model.input_layer.out_features == 32
        assert len(model.hidden_layers) == 2
        assert model.output_layer.in_features == 32
        assert model.output_layer.out_features == 2

    def test_forward(self):
        """Test forward pass"""
        model = MLP(input_dim=10, n_hidden_layers=2, hidden_dim=32, output_dim=2)
        x = torch.randn(32, 10)

        output = model(x)

        assert output.shape == (32, 2)

    def test_zero_hidden_layers(self):
        """Test with zero hidden layers"""
        model = MLP(input_dim=10, n_hidden_layers=0, hidden_dim=32, output_dim=2)
        x = torch.randn(16, 10)

        output = model(x)

        assert output.shape == (16, 2)
        assert len(model.hidden_layers) == 0

    def test_deep_network(self):
        """Test with many hidden layers"""
        model = MLP(input_dim=10, n_hidden_layers=5, hidden_dim=64, output_dim=3)
        x = torch.randn(8, 10)

        output = model(x)

        assert output.shape == (8, 3)
        assert len(model.hidden_layers) == 5

    def test_gradient_flows(self):
        """Test that gradients flow through all layers"""
        model = MLP(input_dim=10, n_hidden_layers=2, hidden_dim=32, output_dim=2)
        x = torch.randn(8, 10)
        target = torch.randint(0, 2, (8,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        assert model.input_layer.weight.grad is not None
        assert model.output_layer.weight.grad is not None
        for layer in model.hidden_layers:
            assert layer.weight.grad is not None

    def test_binary_output(self):
        """Test with binary classification output"""
        model = MLP(input_dim=10, n_hidden_layers=1, hidden_dim=16, output_dim=1)
        x = torch.randn(16, 10)

        output = model(x)

        # With output_dim=1, squeeze removes the dimension
        assert output.shape == (16,)
