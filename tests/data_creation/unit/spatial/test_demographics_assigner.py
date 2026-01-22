"""
Tests for Demographics Assigner
"""

import pytest
import networkx as nx
import numpy as np
import os
from src.data_creation.spatial_simulation.demographics_assigner import (
    DemographicsAssigner,
    assign_kyc_from_demographics
)


# Path to test demographics CSV
DEMOGRAPHICS_CSV = os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'parameters', 'small_test', 'demographics.csv'
)


@pytest.fixture
def simple_graph():
    """Create a simple test graph"""
    g = nx.DiGraph()
    # Add 20 nodes with varying degrees
    for i in range(20):
        g.add_node(i)

    # Add edges to create structure
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),  # Node 0 is a hub
        (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9),
        (10, 11), (11, 12), (12, 13),
        (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)
    ]
    g.add_edges_from(edges)
    return g


class TestDemographicsAssigner:

    def test_initialization(self, simple_graph):
        """Test that assigner initializes correctly"""
        assigner = DemographicsAssigner(
            graph=simple_graph,
            demographics_csv=DEMOGRAPHICS_CSV,
            seed=42
        )
        assert len(assigner.demographics) > 0
        assert 'age' in assigner.demographics.columns
        assert 'population size' in assigner.demographics.columns

    def test_assign_creates_attributes(self, simple_graph):
        """Test that assign() creates age, salary, balance attributes"""
        assigner = DemographicsAssigner(
            graph=simple_graph,
            demographics_csv=DEMOGRAPHICS_CSV,
            seed=42
        )
        assigner.assign()

        for node in simple_graph.nodes():
            assert 'age' in simple_graph.nodes[node]
            assert 'salary' in simple_graph.nodes[node]
            assert 'init_balance' in simple_graph.nodes[node]

    def test_age_range(self, simple_graph):
        """Test that ages are within reasonable range"""
        assigner = DemographicsAssigner(
            graph=simple_graph,
            demographics_csv=DEMOGRAPHICS_CSV,
            seed=42
        )
        assigner.assign()

        ages = [simple_graph.nodes[n]['age'] for n in simple_graph.nodes()]
        assert min(ages) >= 16
        assert max(ages) <= 100

    def test_salary_positive(self, simple_graph):
        """Test that salaries are positive"""
        assigner = DemographicsAssigner(
            graph=simple_graph,
            demographics_csv=DEMOGRAPHICS_CSV,
            seed=42
        )
        assigner.assign()

        for node in simple_graph.nodes():
            assert simple_graph.nodes[node]['salary'] > 0

    def test_balance_positive(self, simple_graph):
        """Test that balances are positive"""
        assigner = DemographicsAssigner(
            graph=simple_graph,
            demographics_csv=DEMOGRAPHICS_CSV,
            seed=42
        )
        assigner.assign()

        for node in simple_graph.nodes():
            assert simple_graph.nodes[node]['init_balance'] > 0

    def test_reproducibility(self, simple_graph):
        """Test that same seed produces same results"""
        assigner1 = DemographicsAssigner(
            graph=simple_graph.copy(),
            demographics_csv=DEMOGRAPHICS_CSV,
            seed=42
        )
        assigner1.assign()

        assigner2 = DemographicsAssigner(
            graph=simple_graph.copy(),
            demographics_csv=DEMOGRAPHICS_CSV,
            seed=42
        )
        assigner2.assign()

        for node in simple_graph.nodes():
            assert assigner1.g.nodes[node]['age'] == assigner2.g.nodes[node]['age']
            assert assigner1.g.nodes[node]['salary'] == assigner2.g.nodes[node]['salary']

    def test_balance_correlated_with_salary(self, simple_graph):
        """Test that balance correlates with salary"""
        assigner = DemographicsAssigner(
            graph=simple_graph,
            demographics_csv=DEMOGRAPHICS_CSV,
            seed=42
        )
        assigner.assign()

        salaries = [simple_graph.nodes[n]['salary'] for n in simple_graph.nodes()]
        balances = [simple_graph.nodes[n]['init_balance'] for n in simple_graph.nodes()]

        corr = np.corrcoef(np.log1p(salaries), np.log1p(balances))[0, 1]
        # Should have positive correlation
        assert corr > 0.3

    def test_convenience_function(self, simple_graph):
        """Test the convenience function works"""
        assign_kyc_from_demographics(
            graph=simple_graph,
            demographics_csv=DEMOGRAPHICS_CSV,
            seed=42
        )

        for node in simple_graph.nodes():
            assert 'age' in simple_graph.nodes[node]
            assert 'salary' in simple_graph.nodes[node]
            assert 'init_balance' in simple_graph.nodes[node]

    def test_missing_csv_raises_error(self, simple_graph):
        """Test that missing CSV raises error"""
        with pytest.raises(Exception):
            DemographicsAssigner(
                graph=simple_graph,
                demographics_csv="/nonexistent/path.csv",
                seed=42
            )
