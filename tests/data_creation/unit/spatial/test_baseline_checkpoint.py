"""
Tests for baseline checkpoint save/load functionality.
"""

import pytest
import os
import tempfile
import networkx as nx
from collections import defaultdict

from src.data_creation.spatial_simulation.transaction_graph_generator import (
    TransactionGenerator,
    generate_baseline,
    inject_alerts_from_baseline,
)


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal config for testing checkpoint functionality."""
    # Create input directory
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create minimal degree.csv
    degree_file = input_dir / "degree.csv"
    degree_file.write_text("count,in_degree,out_degree\n10,1,1\n")

    # Create minimal accounts.csv (aggregated format)
    accounts_file = input_dir / "accounts.csv"
    accounts_file.write_text("count,bank_id\n10,bank1\n")

    # Create minimal alert_patterns.csv
    alert_file = input_dir / "alert_patterns.csv"
    alert_file.write_text("count,type,schedule_id,min_accounts,max_accounts,min_period,max_period,bank_id,is_sar,source_type\n")

    # Create minimal normal_models.csv
    normal_models_file = input_dir / "normal_models.csv"
    normal_models_file.write_text("count,type,min_accounts,max_accounts,bank_id\n")

    # Create minimal transaction_type.csv
    tx_type_file = input_dir / "transaction_type.csv"
    tx_type_file.write_text("type,count\nTRANSFER,1\n")

    # Create minimal demographics.csv (using single-age format)
    demographics_file = input_dir / "demographics.csv"
    demographics_file.write_text(
        "age, average year income (tkr), median year income (tkr), population size\n"
        "25, 200.0, 180.0, 10000.0\n"
        "35, 350.0, 320.0, 15000.0\n"
        "45, 450.0, 400.0, 12000.0\n"
        "55, 420.0, 380.0, 8000.0\n"
    )

    config = {
        'general': {
            'simulation_name': 'test_checkpoint',
            'random_seed': 42,
            'total_steps': 100,
        },
        'default': {
            'min_amount': 100,
            'max_amount': 10000,
            'min_balance': 1000,
            'max_balance': 100000,
            'bank_id': 'bank1',
        },
        'input': {
            'directory': str(input_dir),
            'accounts': 'accounts.csv',
            'alert_patterns': 'alert_patterns.csv',
            'normal_models': 'normal_models.csv',
            'degree': 'degree.csv',
            'transaction_type': 'transaction_type.csv',
            'is_aggregated_accounts': True,
        },
        'temporal': {
            'directory': str(output_dir),
            'transactions': 'transactions.csv',
            'accounts': 'accounts.csv',
            'alert_members': 'alert_members.csv',
            'normal_models': 'normal_models.csv',
        },
        'demographics': {
            'csv_path': str(demographics_file),
        },
        'ml_selector': {
            'enabled': False,  # Disable ML selector for simple tests
        },
    }

    return config


class TestBaselineCheckpoint:
    """Tests for checkpoint save/load functionality."""

    def test_save_checkpoint_creates_file(self, minimal_config, tmp_path):
        """Test that save_baseline_checkpoint creates a checkpoint file."""
        txg = TransactionGenerator(minimal_config)
        txg.set_num_accounts()
        txg.generate_normal_transactions()
        txg.load_account_list()

        checkpoint_path = tmp_path / "test_checkpoint.pkl"
        saved_path = txg.save_baseline_checkpoint(str(checkpoint_path))

        assert os.path.exists(saved_path)
        assert saved_path == str(checkpoint_path)

    def test_save_checkpoint_default_path(self, minimal_config):
        """Test that save_baseline_checkpoint uses default path when not specified."""
        txg = TransactionGenerator(minimal_config)
        txg.set_num_accounts()
        txg.generate_normal_transactions()
        txg.load_account_list()

        saved_path = txg.save_baseline_checkpoint()

        expected_path = os.path.join(minimal_config['input']['directory'], 'baseline_checkpoint.pkl')
        assert saved_path == expected_path
        assert os.path.exists(saved_path)

    def test_load_checkpoint_restores_graph(self, minimal_config, tmp_path):
        """Test that load_baseline_checkpoint restores graph state."""
        # Create and save checkpoint
        txg1 = TransactionGenerator(minimal_config)
        txg1.set_num_accounts()
        txg1.generate_normal_transactions()
        txg1.load_account_list()

        # Record state before saving
        num_nodes = txg1.g.number_of_nodes()
        num_edges = txg1.g.number_of_edges()

        checkpoint_path = tmp_path / "test_checkpoint.pkl"
        txg1.save_baseline_checkpoint(str(checkpoint_path))

        # Create new generator and load checkpoint
        txg2 = TransactionGenerator(minimal_config)
        txg2.load_baseline_checkpoint(str(checkpoint_path))

        # Verify state is restored
        assert txg2.g.number_of_nodes() == num_nodes
        assert txg2.g.number_of_edges() == num_edges
        assert txg2.num_accounts == txg1.num_accounts

    def test_load_checkpoint_restores_bank_mappings(self, minimal_config, tmp_path):
        """Test that bank mappings are correctly restored."""
        # Create and save checkpoint
        txg1 = TransactionGenerator(minimal_config)
        txg1.set_num_accounts()
        txg1.generate_normal_transactions()
        txg1.load_account_list()

        checkpoint_path = tmp_path / "test_checkpoint.pkl"
        txg1.save_baseline_checkpoint(str(checkpoint_path))

        # Create new generator and load checkpoint
        txg2 = TransactionGenerator(minimal_config)
        txg2.load_baseline_checkpoint(str(checkpoint_path))

        # Verify bank mappings
        assert dict(txg2.bank_to_accts) == dict(txg1.bank_to_accts)
        assert txg2.acct_to_bank == txg1.acct_to_bank

    def test_load_checkpoint_resets_alert_groups(self, minimal_config, tmp_path):
        """Test that alert_groups is reset when loading checkpoint."""
        # Create and save checkpoint
        txg1 = TransactionGenerator(minimal_config)
        txg1.set_num_accounts()
        txg1.generate_normal_transactions()
        txg1.load_account_list()

        # Manually add some alert groups to simulate previous state
        txg1.alert_groups = {1: nx.DiGraph(), 2: nx.DiGraph()}

        checkpoint_path = tmp_path / "test_checkpoint.pkl"
        txg1.save_baseline_checkpoint(str(checkpoint_path))

        # Create new generator with some alert groups
        txg2 = TransactionGenerator(minimal_config)
        txg2.alert_groups = {99: nx.DiGraph()}

        # Load checkpoint
        txg2.load_baseline_checkpoint(str(checkpoint_path))

        # Alert groups should be reset to empty
        assert txg2.alert_groups == {}

    def test_load_checkpoint_resets_ml_selector(self, minimal_config, tmp_path):
        """Test that ml_selector is reset when loading checkpoint."""
        # Create and save checkpoint
        txg1 = TransactionGenerator(minimal_config)
        txg1.set_num_accounts()
        txg1.generate_normal_transactions()
        txg1.load_account_list()

        checkpoint_path = tmp_path / "test_checkpoint.pkl"
        txg1.save_baseline_checkpoint(str(checkpoint_path))

        # Create new generator
        txg2 = TransactionGenerator(minimal_config)
        txg2.ml_selector = "dummy_selector"  # Set a dummy value

        # Load checkpoint
        txg2.load_baseline_checkpoint(str(checkpoint_path))

        # ML selector should be reset to None
        assert txg2.ml_selector is None

    def test_checkpoint_preserves_edge_id_counter(self, minimal_config, tmp_path):
        """Test that edge_id counter is preserved through checkpoint."""
        # Create and save checkpoint
        txg1 = TransactionGenerator(minimal_config)
        txg1.set_num_accounts()
        txg1.generate_normal_transactions()
        txg1.load_account_list()

        original_edge_id = txg1.edge_id

        checkpoint_path = tmp_path / "test_checkpoint.pkl"
        txg1.save_baseline_checkpoint(str(checkpoint_path))

        # Create new generator and load checkpoint
        txg2 = TransactionGenerator(minimal_config)
        txg2.load_baseline_checkpoint(str(checkpoint_path))

        # Edge ID should be preserved
        assert txg2.edge_id == original_edge_id

    def test_checkpoint_preserves_hubs(self, minimal_config, tmp_path):
        """Test that hub accounts are preserved through checkpoint."""
        # Create and save checkpoint
        txg1 = TransactionGenerator(minimal_config)
        txg1.set_num_accounts()
        txg1.generate_normal_transactions()
        txg1.load_account_list()
        txg1.set_main_acct_candidates()

        original_hubs = txg1.hubs.copy()

        checkpoint_path = tmp_path / "test_checkpoint.pkl"
        txg1.save_baseline_checkpoint(str(checkpoint_path))

        # Create new generator and load checkpoint
        txg2 = TransactionGenerator(minimal_config)
        txg2.load_baseline_checkpoint(str(checkpoint_path))

        # Hubs should be preserved
        assert txg2.hubs == original_hubs


class TestTwoPhaseGeneration:
    """Tests for two-phase generation functions."""

    def test_generate_baseline_creates_checkpoint(self, minimal_config, tmp_path):
        """Test that generate_baseline creates a checkpoint file."""
        checkpoint_path = tmp_path / "baseline.pkl"

        saved_path = generate_baseline(minimal_config, checkpoint_path=str(checkpoint_path))

        assert os.path.exists(saved_path)
        assert saved_path == str(checkpoint_path)

    def test_inject_alerts_from_baseline_requires_checkpoint(self, minimal_config, tmp_path):
        """Test that inject_alerts_from_baseline raises error if checkpoint missing."""
        checkpoint_path = tmp_path / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            inject_alerts_from_baseline(minimal_config, checkpoint_path=str(checkpoint_path))

    def test_two_phase_flow(self, minimal_config, tmp_path):
        """Test the complete two-phase generation flow."""
        # Add an alert pattern for the second phase
        alert_file = minimal_config['input']['directory'] + '/alert_patterns.csv'
        with open(alert_file, 'w') as f:
            f.write("count,type,schedule_id,min_accounts,max_accounts,min_period,max_period,bank_id,is_sar,source_type\n")
            f.write("1,fan_in,1,3,4,0,50,,true,TRANSFER\n")

        checkpoint_path = tmp_path / "baseline.pkl"

        # Phase 1: Generate baseline
        saved_path = generate_baseline(minimal_config, checkpoint_path=str(checkpoint_path))
        assert os.path.exists(saved_path)

        # Phase 2: Inject alerts from baseline
        txg = inject_alerts_from_baseline(minimal_config, checkpoint_path=str(checkpoint_path))

        # Verify alerts were injected
        assert len(txg.alert_groups) > 0

    def test_two_phase_allows_different_configs(self, minimal_config, tmp_path):
        """Test that different configs can be used for alert injection."""
        checkpoint_path = tmp_path / "baseline.pkl"

        # Phase 1: Generate baseline
        generate_baseline(minimal_config, checkpoint_path=str(checkpoint_path))

        # Create modified config with ML selector enabled
        modified_config = minimal_config.copy()
        modified_config['ml_selector'] = {
            'enabled': True,
            'n_target_labels': 1,
            'n_seeds_per_target': 1,
            'restart_alpha': 0.15,
            'structure_weights': {'degree': 0.5, 'betweenness': 0.3, 'pagerank': 0.2},
            'kyc_weights': {'balance': 0.4, 'salary': 0.3, 'age': 0.3},
            'participation_decay': 0.5,
        }

        # Add alert pattern
        alert_file = minimal_config['input']['directory'] + '/alert_patterns.csv'
        with open(alert_file, 'w') as f:
            f.write("count,type,schedule_id,min_accounts,max_accounts,min_period,max_period,bank_id,is_sar,source_type\n")
            f.write("1,fan_out,1,3,4,0,50,,true,TRANSFER\n")

        # Phase 2: Inject alerts with different config
        txg = inject_alerts_from_baseline(modified_config, checkpoint_path=str(checkpoint_path))

        # ML selector should be prepared with the new config
        assert txg.ml_selector is not None
        assert len(txg.alert_groups) > 0
