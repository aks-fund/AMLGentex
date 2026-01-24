"""
Unit tests for ml/models/losses.py
"""
import pytest
import numpy as np
import torch

from src.ml.models.losses import ClassBalancedLoss, DAMLoss


@pytest.mark.unit
class TestClassBalancedLoss:
    """Tests for ClassBalancedLoss"""

    def test_init(self):
        """Test initialization"""
        n_samples = np.array([100, 10])  # Imbalanced classes
        loss_fn = ClassBalancedLoss(gamma=0.99, n_samples_per_classes=n_samples)

        assert loss_fn.gamma == 0.99
        assert loss_fn.n_classes == 2
        assert loss_fn.loss_type == 'sigmoid'

    def test_forward_sigmoid(self):
        """Test forward pass with sigmoid loss"""
        n_samples = np.array([100, 10])
        loss_fn = ClassBalancedLoss(gamma=0.99, n_samples_per_classes=n_samples, loss_type='sigmoid')

        logits = torch.randn(8, 2)
        labels = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()  # Scalar
        assert loss.item() > 0  # Loss should be positive

    def test_forward_softmax(self):
        """Test forward pass with softmax loss"""
        n_samples = np.array([100, 10])
        loss_fn = ClassBalancedLoss(gamma=0.99, n_samples_per_classes=n_samples, loss_type='sofmax')  # Note: typo in original

        logits = torch.randn(8, 2)
        labels = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_invalid_loss_type(self):
        """Test that invalid loss type raises error"""
        n_samples = np.array([100, 10])
        loss_fn = ClassBalancedLoss(gamma=0.99, n_samples_per_classes=n_samples, loss_type='invalid')

        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])

        with pytest.raises(ValueError, match="loss_type must be sigmoid or softmax"):
            loss_fn(logits, labels)

    def test_multiclass(self):
        """Test with multiple classes"""
        n_samples = np.array([100, 50, 10])  # 3 classes
        loss_fn = ClassBalancedLoss(gamma=0.99, n_samples_per_classes=n_samples)

        logits = torch.randn(6, 3)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flows(self):
        """Test that gradients flow through the loss"""
        n_samples = np.array([100, 10])
        loss_fn = ClassBalancedLoss(gamma=0.99, n_samples_per_classes=n_samples)

        logits = torch.randn(4, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


@pytest.mark.unit
class TestDAMLoss:
    """Tests for DAMLoss (Distribution-Aware Margin Loss)"""

    def test_init(self):
        """Test initialization"""
        class_counts = torch.tensor([100, 10])
        loss_fn = DAMLoss(class_counts, scale=1.0)

        assert loss_fn.margins is not None
        assert loss_fn.margins.shape == (2,)

    def test_margins_inversely_proportional(self):
        """Test that margins are larger for minority classes"""
        class_counts = torch.tensor([100.0, 10.0])
        loss_fn = DAMLoss(class_counts, scale=1.0)

        # Minority class (10 samples) should have larger margin
        assert loss_fn.margins[1] > loss_fn.margins[0]

    def test_scale_affects_margins(self):
        """Test that scale parameter affects margins"""
        class_counts = torch.tensor([100.0, 10.0])
        loss_fn1 = DAMLoss(class_counts, scale=1.0)
        loss_fn2 = DAMLoss(class_counts, scale=2.0)

        # Higher scale should produce larger margins
        assert torch.all(loss_fn2.margins > loss_fn1.margins)

    def test_forward(self):
        """Test forward pass"""
        class_counts = torch.tensor([100.0, 10.0])
        loss_fn = DAMLoss(class_counts, scale=1.0)

        logits = torch.randn(8, 2)
        targets = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0])

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_gradient_flows(self):
        """Test that gradients flow through the loss"""
        class_counts = torch.tensor([100.0, 10.0])
        loss_fn = DAMLoss(class_counts, scale=1.0)

        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
