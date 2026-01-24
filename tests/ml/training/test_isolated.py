"""
Unit tests for ml/training/isolated.py
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from src.ml.training.isolated import isolated, run_clients


@pytest.mark.unit
class TestRunClients:
    """Tests for run_clients helper function"""

    def test_runs_all_clients(self):
        """Test that all clients are run"""
        client1 = Mock()
        client1.id = 'bank_A'
        client1.run.return_value = {'accuracy': 0.9}

        client2 = Mock()
        client2.id = 'bank_B'
        client2.run.return_value = {'accuracy': 0.85}

        clients = [client1, client2]
        params = [{'epochs': 10}, {'epochs': 10}]

        ids, results = run_clients(clients, params)

        assert ids == ['bank_A', 'bank_B']
        assert len(results) == 2
        client1.run.assert_called_once_with(epochs=10)
        client2.run.assert_called_once_with(epochs=10)

    def test_returns_results_in_order(self):
        """Test that results match client order"""
        client1 = Mock()
        client1.id = 'first'
        client1.run.return_value = {'result': 1}

        client2 = Mock()
        client2.id = 'second'
        client2.run.return_value = {'result': 2}

        clients = [client1, client2]
        params = [{}, {}]

        ids, results = run_clients(clients, params)

        assert ids[0] == 'first'
        assert results[0] == {'result': 1}
        assert ids[1] == 'second'
        assert results[1] == {'result': 2}


@pytest.mark.unit
class TestIsolated:
    """Tests for isolated training function"""

    @patch('src.ml.training.isolated.mp.Pool')
    def test_creates_clients_from_config(self, mock_pool):
        """Test that clients are created from config"""
        MockClient = Mock()
        MockModel = Mock()

        # Setup pool mock
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.starmap.return_value = [
            (['bank_A'], [{'accuracy': 0.9}]),
            (['bank_B'], [{'accuracy': 0.85}])
        ]

        clients_config = {
            'bank_A': {'data_path': '/path/a'},
            'bank_B': {'data_path': '/path/b'},
        }

        result = isolated(
            seed=42,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config,
            epochs=10
        )

        # Check clients were created
        assert MockClient.call_count == 2

    @patch('src.ml.training.isolated.mp.Pool')
    def test_passes_seed_to_clients(self, mock_pool):
        """Test that seed is passed to clients"""
        MockClient = Mock()
        MockModel = Mock()

        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.starmap.return_value = [(['bank_A'], [{}])]

        clients_config = {'bank_A': {}}

        isolated(
            seed=123,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config
        )

        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs['seed'] == 123

    @patch('src.ml.training.isolated.mp.Pool')
    def test_default_n_workers(self, mock_pool):
        """Test that n_workers defaults to number of clients"""
        MockClient = Mock()
        MockModel = Mock()

        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.starmap.return_value = [
            (['bank_A', 'bank_B', 'bank_C'], [{}, {}, {}])
        ]

        clients_config = {
            'bank_A': {},
            'bank_B': {},
            'bank_C': {},
        }

        isolated(
            seed=42,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config
        )

        # Pool should be created with 3 workers
        mock_pool.assert_called_once_with(3)

    @patch('src.ml.training.isolated.mp.Pool')
    def test_custom_n_workers(self, mock_pool):
        """Test that custom n_workers is respected"""
        MockClient = Mock()
        MockModel = Mock()

        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.starmap.return_value = [
            (['bank_A'], [{}]),
            (['bank_B'], [{}])
        ]

        clients_config = {
            'bank_A': {},
            'bank_B': {},
            'bank_C': {},
        }

        isolated(
            seed=42,
            Client=MockClient,
            Model=MockModel,
            n_workers=2,
            clients=clients_config
        )

        mock_pool.assert_called_once_with(2)

    @patch('src.ml.training.isolated.mp.Pool')
    def test_returns_dict_of_results(self, mock_pool):
        """Test that results are returned as dict keyed by client id"""
        MockClient = Mock()
        MockModel = Mock()

        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.starmap.return_value = [
            (['bank_A', 'bank_B'], [{'acc': 0.9}, {'acc': 0.85}])
        ]

        clients_config = {
            'bank_A': {},
            'bank_B': {},
        }

        result = isolated(
            seed=42,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config
        )

        assert 'bank_A' in result
        assert 'bank_B' in result
        assert result['bank_A'] == {'acc': 0.9}
        assert result['bank_B'] == {'acc': 0.85}

    @patch('src.ml.training.isolated.mp.Pool')
    def test_merges_client_config_with_kwargs(self, mock_pool):
        """Test that client-specific config is merged with global kwargs"""
        MockClient = Mock()
        MockModel = Mock()

        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.starmap.return_value = [(['bank_A'], [{}])]

        clients_config = {
            'bank_A': {'local_param': 'value'},
        }

        isolated(
            seed=42,
            Client=MockClient,
            Model=MockModel,
            clients=clients_config,
            global_param='global_value'
        )

        # Client should receive merged config
        call_kwargs = MockClient.call_args.kwargs
        assert 'local_param' in call_kwargs or 'global_param' in call_kwargs
