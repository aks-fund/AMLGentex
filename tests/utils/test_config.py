"""
Unit tests for configuration utilities
"""
import pytest
import tempfile
from pathlib import Path

from src.utils.config import discover_clients, build_data_paths


class TestDiscoverClients:
    """Tests for discover_clients function"""

    def test_discovers_client_directories(self, tmp_path):
        """Test that client directories are discovered"""
        # Create preprocessed/clients structure
        clients_dir = tmp_path / "clients"
        clients_dir.mkdir(parents=True)
        (clients_dir / "bank_A").mkdir()
        (clients_dir / "bank_B").mkdir()
        (clients_dir / "bank_C").mkdir()

        clients = discover_clients(tmp_path)

        assert clients == ["bank_A", "bank_B", "bank_C"]

    def test_ignores_hidden_directories(self, tmp_path):
        """Test that hidden directories are ignored"""
        clients_dir = tmp_path / "clients"
        clients_dir.mkdir(parents=True)
        (clients_dir / "bank_A").mkdir()
        (clients_dir / ".hidden").mkdir()

        clients = discover_clients(tmp_path)

        assert clients == ["bank_A"]
        assert ".hidden" not in clients

    def test_returns_empty_if_no_clients_dir(self, tmp_path):
        """Test returns empty list if clients directory doesn't exist"""
        clients = discover_clients(tmp_path)
        assert clients == []

    def test_returns_sorted_clients(self, tmp_path):
        """Test that clients are returned in sorted order"""
        clients_dir = tmp_path / "clients"
        clients_dir.mkdir(parents=True)
        (clients_dir / "zebra_bank").mkdir()
        (clients_dir / "alpha_bank").mkdir()
        (clients_dir / "beta_bank").mkdir()

        clients = discover_clients(tmp_path)

        assert clients == ["alpha_bank", "beta_bank", "zebra_bank"]


class TestBuildDataPaths:
    """Tests for build_data_paths function"""

    def test_centralized_paths_without_edges(self, tmp_path):
        """Test centralized paths for non-GNN clients"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchClient",
            setting="centralized"
        )

        assert "trainset_nodes" in paths
        assert "trainset_edges" not in paths
        assert "centralized" in paths["trainset_nodes"]

    def test_centralized_paths_with_edges(self, tmp_path):
        """Test centralized paths for GNN clients"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchGeometricClient",
            setting="centralized"
        )

        assert "trainset_nodes" in paths
        assert "trainset_edges" in paths

    def test_federated_paths(self, tmp_path):
        """Test federated paths with explicit clients"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchClient",
            setting="federated",
            clients=["bank_A", "bank_B"]
        )

        assert "clients" in paths
        assert "bank_A" in paths["clients"]
        assert "bank_B" in paths["clients"]
        assert "trainset_nodes" in paths["clients"]["bank_A"]

    def test_federated_paths_with_edges(self, tmp_path):
        """Test federated paths for GNN clients include edges"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchGeometricClient",
            setting="federated",
            clients=["bank_A"]
        )

        assert "trainset_edges" in paths["clients"]["bank_A"]

    def test_isolated_paths(self, tmp_path):
        """Test isolated paths (same structure as federated)"""
        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchClient",
            setting="isolated",
            clients=["bank_A"]
        )

        assert "clients" in paths
        assert "bank_A" in paths["clients"]

    def test_auto_discovers_clients(self, tmp_path):
        """Test that clients are auto-discovered if not provided"""
        # Create client directories
        clients_dir = tmp_path / "preprocessed" / "clients"
        clients_dir.mkdir(parents=True)
        (clients_dir / "discovered_bank").mkdir()

        paths = build_data_paths(
            experiment_root=tmp_path,
            client_type="TorchClient",
            setting="federated",
            clients=None  # Auto-discover
        )

        assert "discovered_bank" in paths["clients"]
