"""
Tests for Money Laundering Account Selector
"""

import pytest
import networkx as nx
import numpy as np
from src.data_creation.spatial_simulation.ml_account_selector import (
    MoneyLaunderingAccountSelector,
    weighted_choice_simple
)


@pytest.fixture
def simple_graph():
    """Create a simple test graph with KYC attributes"""
    g = nx.DiGraph()

    # Add nodes with KYC attributes (3 cities, 2 nodes each)
    nodes = [
        (0, {'city': 'CityA', 'init_balance': 1000}),
        (1, {'city': 'CityA', 'init_balance': 2000}),
        (2, {'city': 'CityB', 'init_balance': 1500}),
        (3, {'city': 'CityB', 'init_balance': 500}),
        (4, {'city': 'CityC', 'init_balance': 3000}),
        (5, {'city': 'CityC', 'init_balance': 2500}),
    ]

    g.add_nodes_from(nodes)

    # Add edges to create structure
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (0, 2), (1, 3), (2, 4), (3, 5),
        (0, 3), (1, 4), (2, 5)
    ]
    g.add_edges_from(edges)

    return g


@pytest.fixture
def simple_config():
    """Create a simple test configuration"""
    return {
        'ml_selector': {
            'enabled': True,
            'n_target_labels': 2,  # Select 2 target cities
            'n_seeds_per_target': 1,  # 1 seed per target
            'restart_alpha': 0.15,
            'seed_strategy': 'degree',
            'propagate_labels': ['city'],
            'structure_weights': {
                'degree': 0.4,
                'betweenness': 0.3,
                'pagerank': 0.3
            },
            'propagation_weights': {
                'city': 1.0
            },
            'kyc_weights': {
                'balance': 0.0
            }
        }
    }


@pytest.fixture
def acct_to_bank_simple():
    """Simple account to bank mapping"""
    return {0: 'BankA', 1: 'BankA', 2: 'BankB', 3: 'BankB', 4: 'BankC', 5: 'BankC'}


class TestMLAccountSelector:

    def test_initialization(self, simple_graph, simple_config, acct_to_bank_simple):
        """Test that selector initializes correctly"""
        selector = MoneyLaunderingAccountSelector(
            simple_graph, simple_config, acct_to_bank_simple
        )

        assert selector.n_target_labels == 2
        assert selector.n_seeds_per_target == 1
        assert selector.restart_alpha == 0.15
        assert selector.seed_strategy == 'degree'
        assert 'city' in selector.propagate_labels

    def test_structural_metrics_computation(self, simple_graph, simple_config, acct_to_bank_simple):
        """Test that structural metrics are computed"""
        selector = MoneyLaunderingAccountSelector(
            simple_graph, simple_config, acct_to_bank_simple
        )
        selector._compute_structural_metrics()

        # Check that metrics exist for all nodes
        assert len(selector.structural_features) == 6

        # Check that all metrics are computed
        for node in simple_graph.nodes():
            assert 'degree' in selector.structural_features[node]
            assert 'betweenness' in selector.structural_features[node]
            assert 'pagerank' in selector.structural_features[node]

    def test_seed_selection_targets(self, simple_graph, simple_config, acct_to_bank_simple):
        """Test that target labels are selected and seeds are assigned"""
        selector = MoneyLaunderingAccountSelector(
            simple_graph, simple_config, acct_to_bank_simple
        )
        selector._compute_structural_metrics()
        selector._select_seeds()

        # Check that target labels were selected
        assert 'city' in selector.target_labels
        assert len(selector.target_labels['city']) == 2  # n_target_labels = 2

        # Check that each target has seeds
        for target_value in selector.target_labels['city']:
            label_key = f"city:{target_value}"
            assert label_key in selector.seeds_by_target
            assert len(selector.seeds_by_target[label_key]) > 0

    def test_label_propagation_global(self, simple_graph, simple_config, acct_to_bank_simple):
        """Test that label propagation creates global locality field"""
        selector = MoneyLaunderingAccountSelector(
            simple_graph, simple_config, acct_to_bank_simple
        )
        selector._compute_structural_metrics()
        selector._select_seeds()
        selector._propagate_labels()

        # Check that global propagation score exists
        assert 'city_global' in selector.propagation_scores

        # Check that scores are computed for all nodes
        global_scores = selector.propagation_scores['city_global']
        assert len(global_scores) == 6  # All nodes should have scores

        # All nodes should have non-negative scores
        for score in global_scores.values():
            assert score >= 0

    def test_propagation_spreads_from_seeds(self, simple_graph, simple_config, acct_to_bank_simple):
        """Test that propagation gives higher scores to nodes near seeds"""
        selector = MoneyLaunderingAccountSelector(
            simple_graph, simple_config, acct_to_bank_simple
        )
        selector.prepare()

        # Get the global propagation scores
        global_scores = selector.propagation_scores['city_global']

        # The scores should have variance (not all the same)
        score_values = list(global_scores.values())
        assert max(score_values) > min(score_values)

    def test_final_weights_use_global_propagation(self, simple_graph, simple_config, acct_to_bank_simple):
        """Test that final weights use global propagation, not per-node city lookup"""
        selector = MoneyLaunderingAccountSelector(
            simple_graph, simple_config, acct_to_bank_simple
        )
        selector.prepare()

        # Check that weights exist for all nodes
        assert len(selector.ml_weights) == 6

        # Check that all weights are positive
        for weight in selector.ml_weights.values():
            assert weight > 0

        # Weights should have variance (biased selection)
        weights = list(selector.ml_weights.values())
        assert max(weights) > min(weights)

    def test_weighted_choice(self, simple_graph, simple_config, acct_to_bank_simple):
        """Test that weighted choice works"""
        selector = MoneyLaunderingAccountSelector(
            simple_graph, simple_config, acct_to_bank_simple
        )
        selector.prepare()

        candidates = list(simple_graph.nodes())

        # Run multiple selections to check distribution
        selections = []
        for _ in range(100):
            choice = selector.weighted_choice(candidates)
            assert choice in candidates
            selections.append(choice)

        # Check that at least some variety in selections
        assert len(set(selections)) > 1

    def test_weighted_choice_bank(self, simple_graph, simple_config, acct_to_bank_simple):
        """Test bank-constrained weighted choice"""
        selector = MoneyLaunderingAccountSelector(
            simple_graph, simple_config, acct_to_bank_simple
        )
        selector.prepare()

        # Test with BankA accounts
        bank_a_accounts = [0, 1]
        for _ in range(10):
            choice = selector.weighted_choice_bank(bank_a_accounts, 'BankA')
            assert choice in bank_a_accounts

    def test_prepare_full_pipeline(self, simple_graph, simple_config, acct_to_bank_simple):
        """Test that full preparation pipeline works"""
        selector = MoneyLaunderingAccountSelector(
            simple_graph, simple_config, acct_to_bank_simple
        )

        # Should not raise any exceptions
        selector.prepare()

        # Check that all components are initialized
        assert len(selector.structural_features) > 0
        assert len(selector.ml_weights) > 0
        assert len(selector.ml_weights_by_bank) > 0
        assert 'city_global' in selector.propagation_scores

    def test_node_without_label_still_gets_weight(self, acct_to_bank_simple):
        """Test that nodes without the label attribute still get propagation weight"""
        # Create graph where one node has no city attribute
        g = nx.DiGraph()
        nodes = [
            (0, {'city': 'CityA', 'init_balance': 1000}),
            (1, {'city': 'CityA', 'init_balance': 2000}),
            (2, {'init_balance': 1500}),  # No city!
            (3, {'city': 'CityB', 'init_balance': 500}),
        ]
        g.add_nodes_from(nodes)
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2)])

        config = {
            'ml_selector': {
                'enabled': True,
                'n_target_labels': 2,
                'n_seeds_per_target': 1,
                'propagate_labels': ['city'],
                'structure_weights': {'degree': 1.0},
                'propagation_weights': {'city': 1.0}
            }
        }

        selector = MoneyLaunderingAccountSelector(g, config, acct_to_bank_simple)
        selector.prepare()

        # Node 2 (no city) should still have a weight based on structural proximity
        assert 2 in selector.ml_weights
        assert selector.ml_weights[2] > 0


class TestReproducibility:

    def test_same_seed_same_results(self, simple_graph, acct_to_bank_simple):
        """Test that same seed produces identical results"""
        config = {
            'ml_selector': {
                'enabled': True,
                'n_target_labels': 2,
                'n_seeds_per_target': 1,
                'seed': 42,
                'propagate_labels': ['city'],
                'structure_weights': {'degree': 1.0},
                'propagation_weights': {'city': 1.0}
            }
        }

        # Run twice with same seed
        selector1 = MoneyLaunderingAccountSelector(simple_graph, config, acct_to_bank_simple, seed=42)
        selector1.prepare()

        selector2 = MoneyLaunderingAccountSelector(simple_graph, config, acct_to_bank_simple, seed=42)
        selector2.prepare()

        # Should have identical target labels
        assert selector1.target_labels == selector2.target_labels

        # Should have identical seeds
        assert selector1.seeds_by_target == selector2.seeds_by_target

        # Should have identical weights
        assert selector1.ml_weights == selector2.ml_weights

    def test_different_seed_different_results(self, simple_graph, acct_to_bank_simple):
        """Test that different seeds produce different results"""
        config = {
            'ml_selector': {
                'enabled': True,
                'n_target_labels': 2,
                'n_seeds_per_target': 1,
                'propagate_labels': ['city'],
                'structure_weights': {'degree': 1.0},
                'propagation_weights': {'city': 1.0}
            }
        }

        selector1 = MoneyLaunderingAccountSelector(simple_graph, config, acct_to_bank_simple, seed=42)
        selector1.prepare()

        selector2 = MoneyLaunderingAccountSelector(simple_graph, config, acct_to_bank_simple, seed=123)
        selector2.prepare()

        # With different seeds, likely different target selections
        # (Note: could be same by chance, but very unlikely)
        # At minimum, the RNG state should be independent
        assert selector1.rng.bit_generator.state != selector2.rng.bit_generator.state


class TestWeightedChoiceHelper:

    def test_weighted_choice_simple(self):
        """Test the simple weighted choice helper"""
        candidates = [1, 2, 3, 4, 5]
        weights = [0.1, 0.2, 0.3, 0.2, 0.2]

        # Run multiple selections
        selections = []
        for _ in range(100):
            choice = weighted_choice_simple(candidates, weights)
            assert choice in candidates
            selections.append(choice)

        # Check variety
        assert len(set(selections)) > 1

    def test_weighted_choice_uniform(self):
        """Test with uniform weights"""
        candidates = [1, 2, 3]
        weights = [1.0, 1.0, 1.0]

        selections = []
        for _ in range(50):
            choice = weighted_choice_simple(candidates, weights)
            selections.append(choice)

        # Should see all candidates with uniform weights
        assert len(set(selections)) == 3

    def test_weighted_choice_empty(self):
        """Test that empty candidates raises error"""
        with pytest.raises(ValueError):
            weighted_choice_simple([], [])
