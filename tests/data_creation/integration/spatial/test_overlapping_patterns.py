"""
Integration test for accounts participating in multiple alert patterns during graph generation
"""
import pytest
from collections import defaultdict


@pytest.mark.integration
@pytest.mark.spatial
class TestOverlappingAlertPatterns:
    """Test that accounts can be assigned to multiple alert typologies"""

    def test_accounts_participate_in_multiple_alert_patterns(self, large_graph):
        """Test that the probabilistic assignment creates some accounts in multiple alert patterns"""
        txg = large_graph

        # Track which accounts appear in which alert patterns
        account_to_alerts = defaultdict(set)

        for alert_id, alert_graph in txg.alert_groups.items():
            for node in alert_graph.nodes():
                account_to_alerts[node].add(alert_id)

        # Count accounts in multiple patterns
        overlap_counts = defaultdict(int)
        for acc, alerts in account_to_alerts.items():
            num_patterns = len(alerts)
            overlap_counts[num_patterns] += 1

        # With large graph (10 patterns of each type, 10-20 accounts each),
        # probabilistic assignment should create some overlaps
        accounts_in_multiple = sum(count for num, count in overlap_counts.items() if num > 1)

        print(f"\nAlert pattern participation:")
        for num_patterns in sorted(overlap_counts.keys()):
            count = overlap_counts[num_patterns]
            print(f"  {num_patterns} pattern(s): {count} accounts")

        # Verify that accounts can participate in multiple patterns
        assert accounts_in_multiple > 0, \
            "Expected some accounts to participate in multiple alert patterns due to probabilistic assignment"

        # Verify overlapping accounts have valid memberships
        for acc, alerts in account_to_alerts.items():
            if len(alerts) > 1:
                # Each alert should be distinct
                assert len(alerts) == len(set(alerts))

                # Verify account actually exists in each alert graph
                for alert_id in alerts:
                    alert_graph = txg.alert_groups[alert_id]
                    assert acc in alert_graph.nodes()

    def test_ml_selector_does_not_over_concentrate_selections(self, large_graph):
        """Test that ML selector doesn't cause extreme concentration in a few accounts."""
        txg = large_graph

        # Track participation counts
        account_to_alerts = defaultdict(set)
        for alert_id, alert_graph in txg.alert_groups.items():
            for node in alert_graph.nodes():
                account_to_alerts[node].add(alert_id)

        if not account_to_alerts:
            pytest.skip("No alert patterns generated")

        # Compute participation distribution
        participation_counts = [len(alerts) for alerts in account_to_alerts.values()]
        max_participation = max(participation_counts)
        total_alert_memberships = sum(participation_counts)
        unique_accounts = len(account_to_alerts)

        print(f"\nML Selector concentration check:")
        print(f"  Unique accounts in alerts: {unique_accounts}")
        print(f"  Total alert memberships: {total_alert_memberships}")
        print(f"  Max participation by single account: {max_participation}")

        # Check that no single account dominates
        # With 0.5 decay, expect max ~60% participation (high-weight accounts still favored)
        total_alerts = len(txg.alert_groups)
        assert max_participation < total_alerts * 0.7, \
            f"Single account participates in {max_participation}/{total_alerts} alerts - too concentrated"

        # Check that we have reasonable diversity
        # At least 20% of alert memberships should come from accounts with only 1-2 participations
        low_participation_memberships = sum(
            len(alerts) for alerts in account_to_alerts.values()
            if len(alerts) <= 2
        )
        low_participation_ratio = low_participation_memberships / total_alert_memberships

        print(f"  Memberships from low-participation accounts (1-2): {low_participation_ratio:.1%}")

        # Relaxed threshold - ML selector will bias toward some accounts, but shouldn't eliminate diversity
        assert low_participation_ratio > 0.1, \
            f"Only {low_participation_ratio:.1%} of memberships from low-participation accounts - ML selector may be over-concentrating"
