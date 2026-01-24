"""
Unit tests for ml/training/federated.py
"""
import pytest
from unittest.mock import Mock, MagicMock

from src.ml.training.federated import federated


@pytest.mark.unit
class TestFederated:
    """Tests for federated training function"""

    def test_creates_clients_from_config(self):
        """Test that clients are created from config"""
        MockClient = Mock()
        MockServer = Mock()
        MockServer.return_value.run.return_value = {'accuracy': 0.9}
        MockModel = Mock()

        clients_config = {
            'bank_A': {'data_path': '/path/a'},
            'bank_B': {'data_path': '/path/b'},
        }

        result = federated(
            seed=42,
            Server=MockServer,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config,
            epochs=10
        )

        # Check clients were created
        assert MockClient.call_count == 2

        # Check server was created with clients
        MockServer.assert_called_once()
        server_call_kwargs = MockServer.call_args.kwargs
        assert len(server_call_kwargs['clients']) == 2

    def test_passes_seed_to_clients(self):
        """Test that seed is passed to clients"""
        MockClient = Mock()
        MockServer = Mock()
        MockServer.return_value.run.return_value = {}
        MockModel = Mock()

        clients_config = {'bank_A': {}}

        federated(
            seed=123,
            Server=MockServer,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config
        )

        # Check seed was passed
        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs['seed'] == 123

    def test_passes_model_to_clients(self):
        """Test that Model class is passed to clients"""
        MockClient = Mock()
        MockServer = Mock()
        MockServer.return_value.run.return_value = {}
        MockModel = Mock()

        clients_config = {'bank_A': {}}

        federated(
            seed=42,
            Server=MockServer,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config
        )

        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs['Model'] == MockModel

    def test_default_n_workers(self):
        """Test that n_workers defaults to number of clients"""
        MockClient = Mock()
        MockServer = Mock()
        MockServer.return_value.run.return_value = {}
        MockModel = Mock()

        clients_config = {
            'bank_A': {},
            'bank_B': {},
            'bank_C': {},
        }

        federated(
            seed=42,
            Server=MockServer,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config
        )

        server_call_kwargs = MockServer.call_args.kwargs
        assert server_call_kwargs['n_workers'] == 3

    def test_custom_n_workers(self):
        """Test that custom n_workers is respected"""
        MockClient = Mock()
        MockServer = Mock()
        MockServer.return_value.run.return_value = {}
        MockModel = Mock()

        clients_config = {
            'bank_A': {},
            'bank_B': {},
            'bank_C': {},
        }

        federated(
            seed=42,
            Server=MockServer,
            Client=MockClient,
            Model=MockModel,
            n_workers=2,
            clients=clients_config
        )

        server_call_kwargs = MockServer.call_args.kwargs
        assert server_call_kwargs['n_workers'] == 2

    def test_returns_server_results(self):
        """Test that server results are returned"""
        MockClient = Mock()
        MockServer = Mock()
        expected_results = {'accuracy': 0.95, 'loss': 0.1}
        MockServer.return_value.run.return_value = expected_results
        MockModel = Mock()

        clients_config = {'bank_A': {}}

        result = federated(
            seed=42,
            Server=MockServer,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config
        )

        assert result == expected_results

    def test_merges_client_config_with_kwargs(self):
        """Test that client-specific config is merged with global kwargs"""
        MockClient = Mock()
        MockServer = Mock()
        MockServer.return_value.run.return_value = {}
        MockModel = Mock()

        clients_config = {
            'bank_A': {'local_epochs': 5},  # Client-specific
        }

        federated(
            seed=42,
            Server=MockServer,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config,
            learning_rate=0.01  # Global param
        )

        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs['local_epochs'] == 5
        assert call_kwargs['learning_rate'] == 0.01
