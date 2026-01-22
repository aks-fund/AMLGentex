"""
Integration tests for KYC demographics and ML alert selection.

Tests correlations between:
- Demographics attributes (age, salary, balance)
- Structural features (degree, betweenness, pagerank)
- ML selection weights
- Alert account assignments
"""

import pytest
import numpy as np
import networkx as nx
from scipy import stats
from pathlib import Path

from src.data_creation.spatial_simulation.demographics_assigner import DemographicsAssigner
from src.data_creation.spatial_simulation.ml_account_selector import MoneyLaunderingAccountSelector


def wrap_ml_config(ml_config: dict) -> dict:
    """Wrap ML selector config in expected format."""
    return {'ml_selector': ml_config}


# Path to test demographics CSV
DEMOGRAPHICS_CSV = Path(__file__).parent.parent.parent / 'parameters' / 'small_test' / 'demographics.csv'


@pytest.fixture
def medium_graph():
    """Create a medium-sized graph with realistic structure."""
    # Create scale-free-ish graph with 200 nodes
    g = nx.DiGraph()
    n_nodes = 200

    for i in range(n_nodes):
        g.add_node(i, bank_id=f"bank_{i % 3}", city=f"city_{i % 5}")

    # Add edges with preferential attachment style
    rng = np.random.default_rng(42)
    for i in range(n_nodes):
        # Each node connects to 2-5 others
        n_edges = rng.integers(2, 6)
        # Bias towards lower-numbered nodes (simulating preferential attachment)
        targets = rng.choice(n_nodes, size=min(n_edges, n_nodes-1), replace=False,
                            p=np.array([1/(j+1) for j in range(n_nodes)]) / sum(1/(j+1) for j in range(n_nodes)))
        for t in targets:
            if t != i:
                g.add_edge(i, t)

    return g


@pytest.fixture
def acct_to_bank(medium_graph):
    """Create account to bank mapping."""
    return {n: medium_graph.nodes[n]['bank_id'] for n in medium_graph.nodes()}


@pytest.mark.integration
@pytest.mark.spatial
class TestDemographicsCorrelations:
    """Test that demographics assignment produces expected correlations."""

    def test_salary_balance_correlation(self, medium_graph):
        """Balance should correlate with salary (controlled by alpha_salary)."""
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42,
            balance_params={
                'balance_months': 2.5,
                'alpha_salary': 0.6,
                'alpha_struct': 0.0,  # Disable struct to isolate salary effect
                'sigma': 0.3
            }
        )
        assigner.assign()

        salaries = [medium_graph.nodes[n]['salary'] for n in medium_graph.nodes()]
        balances = [medium_graph.nodes[n]['init_balance'] for n in medium_graph.nodes()]

        # Log-log correlation should be strong
        corr, p_value = stats.pearsonr(np.log1p(salaries), np.log1p(balances))
        assert corr > 0.5, f"Salary-balance correlation too weak: {corr}"
        assert p_value < 0.01, f"Correlation not significant: p={p_value}"

    def test_structural_balance_correlation(self, medium_graph):
        """Balance should correlate with structural position (controlled by alpha_struct)."""
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42,
            balance_params={
                'balance_months': 2.5,
                'alpha_salary': 0.0,  # Disable salary to isolate struct effect
                'alpha_struct': 0.6,
                'sigma': 0.3
            }
        )
        assigner.assign()

        degrees = [medium_graph.degree(n) for n in medium_graph.nodes()]
        balances = [medium_graph.nodes[n]['init_balance'] for n in medium_graph.nodes()]

        # Log-log correlation should be positive
        corr, p_value = stats.pearsonr(np.log1p(degrees), np.log1p(balances))
        assert corr > 0.3, f"Degree-balance correlation too weak: {corr}"

    def test_age_salary_relationship(self, medium_graph):
        """Salary should vary with age (from demographics data)."""
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        ages = np.array([medium_graph.nodes[n]['age'] for n in medium_graph.nodes()])
        salaries = np.array([medium_graph.nodes[n]['salary'] for n in medium_graph.nodes()])

        # Working-age adults (30-55) should have higher salaries than young/old
        working_age_mask = (ages >= 30) & (ages <= 55)
        young_mask = ages < 25

        if working_age_mask.sum() > 10 and young_mask.sum() > 10:
            working_age_salary = np.median(salaries[working_age_mask])
            young_salary = np.median(salaries[young_mask])
            assert working_age_salary > young_salary, \
                f"Working age salary ({working_age_salary}) should exceed young salary ({young_salary})"

    def test_balance_median_calibration(self, medium_graph):
        """Median balance should be approximately balance_months * median_salary."""
        balance_months = 3.0
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42,
            balance_params={
                'balance_months': balance_months,
                'alpha_salary': 0.6,
                'alpha_struct': 0.4,
                'sigma': 0.3
            }
        )
        assigner.assign()

        balances = [medium_graph.nodes[n]['init_balance'] for n in medium_graph.nodes()]
        median_balance = np.median(balances)

        # Target is balance_months * population_median_salary
        target = balance_months * assigner.median_monthly_salary

        # Should be within 20% of target (some variance from noise)
        ratio = median_balance / target
        assert 0.8 < ratio < 1.2, f"Median balance {median_balance} not close to target {target}"


@pytest.mark.integration
@pytest.mark.spatial
class TestMLSelectorCorrelations:
    """Test that ML selector weights correlate with expected features."""

    def test_weights_correlate_with_degree(self, medium_graph, acct_to_bank):
        """With structure weights, ML weights should correlate with degree."""
        # Assign demographics first
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        config = {
            'n_target_labels': 3,
            'n_seeds_per_target': 5,
            'restart_alpha': 0.15,
            'seed_strategy': 'degree',
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 1.0, 'betweenness': 0.0, 'pagerank': 0.0},
            'propagation_weights': {'city': 0.0},  # Disable propagation
            'kyc_weights': {'init_balance': 0.0, 'salary': 0.0, 'age': 0.0}
        }

        selector = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector.prepare()

        degrees = [medium_graph.degree(n) for n in medium_graph.nodes()]
        weights = [selector.ml_weights.get(n, 0) for n in medium_graph.nodes()]

        # Higher degree should correlate with higher weight
        corr, _ = stats.spearmanr(degrees, weights)
        assert corr > 0.3, f"Degree-weight correlation too weak: {corr}"

    def test_weights_correlate_with_balance(self, medium_graph, acct_to_bank):
        """With KYC weights, ML weights should correlate with balance."""
        # Assign demographics first
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        config = {
            'n_target_labels': 3,
            'n_seeds_per_target': 5,
            'restart_alpha': 0.15,
            'seed_strategy': 'degree',
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 0.0, 'betweenness': 0.0, 'pagerank': 0.0},
            'propagation_weights': {'city': 0.0},
            'kyc_weights': {'init_balance': 1.0, 'salary': 0.0, 'age': 0.0}
        }

        selector = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector.prepare()

        balances = [medium_graph.nodes[n]['init_balance'] for n in medium_graph.nodes()]
        weights = [selector.ml_weights.get(n, 0) for n in medium_graph.nodes()]

        # Higher balance should correlate with higher weight
        corr, _ = stats.spearmanr(balances, weights)
        assert corr > 0.3, f"Balance-weight correlation too weak: {corr}"

    def test_locality_propagation_effect(self, medium_graph, acct_to_bank):
        """Nodes structurally close to seeds should have higher PPR weights."""
        # Assign demographics first
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        # Use only 1 target label with few seeds for clear seed vs non-seed effect
        config = {
            'n_target_labels': 1,
            'n_seeds_per_target': 5,
            'restart_alpha': 0.15,
            'seed_strategy': 'degree',
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 0.0, 'betweenness': 0.0, 'pagerank': 0.0},
            'propagation_weights': {'city': 1.0},  # Only propagation
            'kyc_weights': {'init_balance': 0.0, 'salary': 0.0, 'age': 0.0}
        }

        selector = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector.prepare()

        # Get all seeds
        all_seeds = set()
        for seeds in selector.seeds_by_target.values():
            all_seeds.update(seeds)

        if len(all_seeds) == 0:
            pytest.skip("No seeds selected")

        # Find direct neighbors of seeds (1 hop away)
        undirected = medium_graph.to_undirected()
        seed_neighbors = set()
        for seed in all_seeds:
            seed_neighbors.update(undirected.neighbors(seed))
        seed_neighbors -= all_seeds  # Exclude seeds themselves

        # Find nodes that are NOT neighbors of seeds
        non_neighbor_nodes = set(medium_graph.nodes()) - all_seeds - seed_neighbors

        if len(seed_neighbors) < 5 or len(non_neighbor_nodes) < 5:
            pytest.skip("Not enough neighbor/non-neighbor nodes for comparison")

        # PPR should give higher weights to seed neighbors than distant nodes
        neighbor_weights = [selector.ml_weights.get(n, 0) for n in seed_neighbors]
        non_neighbor_weights = [selector.ml_weights.get(n, 0) for n in non_neighbor_nodes]

        # Seed neighbors should have higher average weight due to PPR propagation
        assert np.mean(neighbor_weights) > np.mean(non_neighbor_weights), \
            f"Seed neighbors (mean={np.mean(neighbor_weights):.6f}) should have higher weights than non-neighbors (mean={np.mean(non_neighbor_weights):.6f})"


@pytest.mark.integration
@pytest.mark.spatial
class TestWeightedSelectionDistribution:
    """Test that weighted selection produces expected distributions."""

    def test_selection_follows_weights(self, medium_graph, acct_to_bank):
        """Selected accounts should be biased towards high-weight nodes."""
        # Assign demographics first
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        config = {
            'n_target_labels': 3,
            'n_seeds_per_target': 5,
            'restart_alpha': 0.15,
            'seed_strategy': 'degree',
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 0.5, 'betweenness': 0.3, 'pagerank': 0.2},
            'propagation_weights': {'city': 0.5},
            'kyc_weights': {'init_balance': 0.3, 'salary': 0.0, 'age': 0.0},
            'participation_decay': 1.0  # Disable decay to test pure weight following
        }

        selector = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector.prepare()

        # Sample many times
        candidates = list(medium_graph.nodes())
        n_samples = 1000
        selections = []

        for _ in range(n_samples):
            selected = selector.weighted_choice(candidates)
            selections.append(selected)

        # Count selections
        selection_counts = {}
        for s in selections:
            selection_counts[s] = selection_counts.get(s, 0) + 1

        # Split nodes into high-weight and low-weight groups
        all_weights = [selector.ml_weights.get(n, 0) for n in candidates]
        weight_threshold = np.median(all_weights)

        high_weight_nodes = [n for n in candidates if selector.ml_weights.get(n, 0) >= weight_threshold]
        low_weight_nodes = [n for n in candidates if selector.ml_weights.get(n, 0) < weight_threshold]

        # Count selections in each group
        high_weight_selections = sum(selection_counts.get(n, 0) for n in high_weight_nodes)
        low_weight_selections = sum(selection_counts.get(n, 0) for n in low_weight_nodes)

        # High-weight nodes should be selected more often
        # Normalize by group size
        high_rate = high_weight_selections / len(high_weight_nodes) if high_weight_nodes else 0
        low_rate = low_weight_selections / len(low_weight_nodes) if low_weight_nodes else 0

        assert high_rate > low_rate, \
            f"High-weight nodes (rate={high_rate:.2f}) should be selected more than low-weight (rate={low_rate:.2f})"

        # Additionally, very high weight nodes should dominate selections
        top_10_pct = int(len(candidates) * 0.1)
        sorted_by_weight = sorted(candidates, key=lambda n: selector.ml_weights.get(n, 0), reverse=True)
        top_nodes = set(sorted_by_weight[:top_10_pct])
        top_selections = sum(1 for s in selections if s in top_nodes)

        # Top 10% by weight should get more than 10% of selections
        assert top_selections / n_samples > 0.10, \
            f"Top 10% by weight should get >10% of selections, got {top_selections/n_samples:.2%}"

    def test_bank_constrained_selection(self, medium_graph, acct_to_bank):
        """Bank-constrained selection should only return nodes from that bank."""
        # Assign demographics first
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        config = {
            'n_target_labels': 3,
            'n_seeds_per_target': 5,
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 0.5, 'betweenness': 0.25, 'pagerank': 0.25},
        }

        selector = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector.prepare()

        # Test bank-constrained selection
        target_bank = "bank_0"
        bank_nodes = [n for n, b in acct_to_bank.items() if b == target_bank]

        for _ in range(100):
            selected = selector.weighted_choice_bank(bank_nodes, target_bank)
            assert acct_to_bank[selected] == target_bank, \
                f"Selected node {selected} not from bank {target_bank}"


@pytest.mark.integration
@pytest.mark.spatial
class TestFullPipelineIntegration:
    """Test the full demographics + ML selector pipeline."""

    def test_full_pipeline_attributes_exist(self, medium_graph, acct_to_bank):
        """After full pipeline, all nodes should have required attributes."""
        # Demographics assignment
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        # ML selector
        config = {
            'n_target_labels': 3,
            'n_seeds_per_target': 5,
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 0.4, 'betweenness': 0.3, 'pagerank': 0.3},
            'propagation_weights': {'city': 0.5},
            'kyc_weights': {'init_balance': 0.2, 'salary': 0.1, 'age': 0.0}
        }

        selector = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector.prepare()

        # Check all nodes have required attributes
        for node in medium_graph.nodes():
            assert 'age' in medium_graph.nodes[node], f"Node {node} missing age"
            assert 'salary' in medium_graph.nodes[node], f"Node {node} missing salary"
            assert 'init_balance' in medium_graph.nodes[node], f"Node {node} missing init_balance"
            assert node in selector.ml_weights, f"Node {node} missing ML weight"

        # Check attribute ranges
        ages = [medium_graph.nodes[n]['age'] for n in medium_graph.nodes()]
        salaries = [medium_graph.nodes[n]['salary'] for n in medium_graph.nodes()]
        balances = [medium_graph.nodes[n]['init_balance'] for n in medium_graph.nodes()]

        assert min(ages) >= 16, "Ages should be >= 16"
        assert max(ages) <= 100, "Ages should be <= 100"
        assert all(s > 0 for s in salaries), "Salaries should be positive"
        assert all(b > 0 for b in balances), "Balances should be positive"

    def test_reproducibility(self, medium_graph, acct_to_bank):
        """Same seed should produce identical results."""
        def run_pipeline(seed):
            g = medium_graph.copy()

            assigner = DemographicsAssigner(
                graph=g,
                demographics_csv=str(DEMOGRAPHICS_CSV),
                seed=seed
            )
            assigner.assign()

            atb = {n: g.nodes[n]['bank_id'] for n in g.nodes()}

            config = {
                'n_target_labels': 3,
                'n_seeds_per_target': 5,
                'propagate_labels': ['city'],
            }

            selector = MoneyLaunderingAccountSelector(
                graph=g,
                config=wrap_ml_config(config),
                acct_to_bank=atb,
                seed=seed
            )
            selector.prepare()

            return (
                [g.nodes[n]['age'] for n in sorted(g.nodes())],
                [g.nodes[n]['salary'] for n in sorted(g.nodes())],
                [selector.ml_weights.get(n, 0) for n in sorted(g.nodes())]
            )

        ages1, salaries1, weights1 = run_pipeline(42)
        ages2, salaries2, weights2 = run_pipeline(42)

        assert ages1 == ages2, "Ages should be reproducible"
        assert salaries1 == salaries2, "Salaries should be reproducible"
        assert weights1 == weights2, "ML weights should be reproducible"

        # Different seed should produce different results
        ages3, salaries3, weights3 = run_pipeline(123)
        assert ages1 != ages3, "Different seeds should produce different ages"


@pytest.mark.integration
@pytest.mark.spatial
class TestAlertMemberStructuralBias:
    """Test that alert members are structurally biased when ML selector is enabled."""

    def test_alert_members_have_higher_degree_when_selector_enabled(self, medium_graph, acct_to_bank):
        """Alert members should have higher average degree when ML selector is enabled."""
        from src.data_creation.spatial_simulation.ml_account_selector import MoneyLaunderingAccountSelector

        # Assign demographics
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        # Configure ML selector with strong degree bias
        config = {
            'n_target_labels': 3,
            'n_seeds_per_target': 5,
            'restart_alpha': 0.15,
            'seed_strategy': 'degree',
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 1.0, 'betweenness': 0.0, 'pagerank': 0.0},
            'propagation_weights': {'city': 0.0},
            'kyc_weights': {'init_balance': 0.0, 'salary': 0.0, 'age': 0.0}
        }

        selector = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector.prepare()

        # Simulate selecting "alert members" using weighted selection
        n_alert_members = 20
        all_nodes = list(medium_graph.nodes())

        weighted_selections = []
        for _ in range(n_alert_members):
            selected = selector.weighted_choice(all_nodes)
            weighted_selections.append(selected)

        # Compare with random selection
        rng = np.random.default_rng(42)
        random_selections = list(rng.choice(all_nodes, size=n_alert_members, replace=True))

        # Compute average degree for each selection method
        weighted_avg_degree = np.mean([medium_graph.degree(n) for n in weighted_selections])
        random_avg_degree = np.mean([medium_graph.degree(n) for n in random_selections])
        population_avg_degree = np.mean([medium_graph.degree(n) for n in all_nodes])

        # Weighted selection should favor higher-degree nodes
        assert weighted_avg_degree > population_avg_degree, \
            f"Weighted selection avg degree ({weighted_avg_degree:.1f}) should exceed population avg ({population_avg_degree:.1f})"

    def test_selection_distribution_shifts_with_structure_weights(self, medium_graph, acct_to_bank):
        """Selection distribution should shift when structure weights change."""
        from src.data_creation.spatial_simulation.ml_account_selector import MoneyLaunderingAccountSelector

        # Assign demographics
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        # Config 1: Degree-only bias
        config_degree = {
            'n_target_labels': 2,
            'n_seeds_per_target': 3,
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 1.0, 'betweenness': 0.0, 'pagerank': 0.0},
            'propagation_weights': {'city': 0.0},
            'kyc_weights': {'init_balance': 0.0, 'salary': 0.0, 'age': 0.0}
        }

        # Config 2: Balance-only bias
        config_balance = {
            'n_target_labels': 2,
            'n_seeds_per_target': 3,
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 0.0, 'betweenness': 0.0, 'pagerank': 0.0},
            'propagation_weights': {'city': 0.0},
            'kyc_weights': {'init_balance': 1.0, 'salary': 0.0, 'age': 0.0}
        }

        selector_degree = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config_degree),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector_degree.prepare()

        selector_balance = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config_balance),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector_balance.prepare()

        # Sample from each
        all_nodes = list(medium_graph.nodes())
        n_samples = 100

        degree_selections = [selector_degree.weighted_choice(all_nodes) for _ in range(n_samples)]
        balance_selections = [selector_balance.weighted_choice(all_nodes) for _ in range(n_samples)]

        # Compute correlations with the biased feature
        degree_corr = stats.spearmanr(
            [medium_graph.degree(n) for n in all_nodes],
            [degree_selections.count(n) for n in all_nodes]
        )[0]

        balance_corr = stats.spearmanr(
            [medium_graph.nodes[n]['init_balance'] for n in all_nodes],
            [balance_selections.count(n) for n in all_nodes]
        )[0]

        # Each selector should favor its configured feature
        # (correlation may be weak due to sampling variance, but should be positive)
        assert degree_corr > 0 or np.isnan(degree_corr), \
            f"Degree selector should positively correlate with degree: {degree_corr}"
        assert balance_corr > 0 or np.isnan(balance_corr), \
            f"Balance selector should positively correlate with balance: {balance_corr}"

    def test_top_weight_nodes_are_selected_more_often(self, medium_graph, acct_to_bank):
        """Top-weighted nodes should be selected disproportionately often."""
        from src.data_creation.spatial_simulation.ml_account_selector import MoneyLaunderingAccountSelector

        # Assign demographics
        assigner = DemographicsAssigner(
            graph=medium_graph,
            demographics_csv=str(DEMOGRAPHICS_CSV),
            seed=42
        )
        assigner.assign()

        config = {
            'n_target_labels': 3,
            'n_seeds_per_target': 5,
            'propagate_labels': ['city'],
            'structure_weights': {'degree': 0.4, 'betweenness': 0.3, 'pagerank': 0.3},
            'propagation_weights': {'city': 0.3},
            'kyc_weights': {'init_balance': 0.2, 'salary': 0.0, 'age': 0.0},
            'participation_decay': 1.0  # Disable decay to test pure weight following
        }

        selector = MoneyLaunderingAccountSelector(
            graph=medium_graph,
            config=wrap_ml_config(config),
            acct_to_bank=acct_to_bank,
            seed=42
        )
        selector.prepare()

        all_nodes = list(medium_graph.nodes())
        n_samples = 500

        # Sample many times
        selections = [selector.weighted_choice(all_nodes) for _ in range(n_samples)]
        selection_counts = {n: selections.count(n) for n in all_nodes}

        # Get top 10% by weight
        sorted_by_weight = sorted(all_nodes, key=lambda n: selector.ml_weights.get(n, 0), reverse=True)
        top_10_pct = set(sorted_by_weight[:int(len(all_nodes) * 0.1)])

        # Count selections from top 10%
        top_10_selections = sum(selection_counts.get(n, 0) for n in top_10_pct)
        top_10_selection_rate = top_10_selections / n_samples

        # Top 10% by weight should get more than 10% of selections
        assert top_10_selection_rate > 0.10, \
            f"Top 10% by weight should get >10% of selections, got {top_10_selection_rate:.1%}"
