"""Tests for post-acceptance attack detection.

This module tests the new post-acceptance attack detection capabilities:
1. AcceptanceTracker state transitions
2. Strategic attacker activation (delayed attacks)
3. Trust trajectory tracking
4. Post-acceptance metrics
"""

import torch

from dfl.config import SimulationConfig
from dfl.verification.acceptance import AcceptanceRecord, AcceptanceStatus, AcceptanceTracker
from dfl.verification.trust import TrustManager


def assert_raises(exception_type, func):
    """Helper to check that a function raises an exception."""
    try:
        func()
        raise AssertionError(f"Expected {exception_type.__name__} to be raised")
    except exception_type:
        pass


class TestAcceptanceTracker:
    """Tests for AcceptanceTracker state machine."""

    def test_initial_state_is_probationary(self):
        """All nodes start as probationary."""
        tracker = AcceptanceTracker(num_nodes=5)

        for node_id in range(5):
            assert tracker.get_status(node_id) == AcceptanceStatus.PROBATIONARY

    def test_acceptance_progression(self):
        """Node progresses to ACCEPTED after clean rounds with sufficient trust."""
        tracker = AcceptanceTracker(
            num_nodes=5,
            acceptance_threshold=0.6,
            consecutive_clean_required=3,
        )

        # Node 0 builds trust over 3 rounds
        for r in range(1, 4):
            changes = tracker.update_round(r, [0.7, 0.5, 0.5, 0.5, 0.5], set())

        assert tracker.get_status(0) == AcceptanceStatus.ACCEPTED
        assert tracker.is_accepted(0)
        assert not tracker.is_post_acceptance_anomaly(0)

    def test_insufficient_trust_blocks_acceptance(self):
        """Node with low trust doesn't achieve acceptance."""
        tracker = AcceptanceTracker(
            num_nodes=5,
            acceptance_threshold=0.6,
            consecutive_clean_required=3,
        )

        # Node 0 has clean rounds but low trust
        for r in range(1, 5):
            tracker.update_round(r, [0.4, 0.5, 0.5, 0.5, 0.5], set())

        # Should still be probationary
        assert tracker.get_status(0) == AcceptanceStatus.PROBATIONARY

    def test_flag_resets_consecutive_count(self):
        """Being flagged resets consecutive clean round count."""
        tracker = AcceptanceTracker(
            num_nodes=5,
            acceptance_threshold=0.6,
            consecutive_clean_required=3,
        )

        # Node 0: 2 clean rounds
        tracker.update_round(1, [0.7, 0.5, 0.5, 0.5, 0.5], set())
        tracker.update_round(2, [0.7, 0.5, 0.5, 0.5, 0.5], set())

        # Node 0 flagged - resets count
        tracker.update_round(3, [0.7, 0.5, 0.5, 0.5, 0.5], {0})

        # 2 more clean rounds - still not enough
        tracker.update_round(4, [0.7, 0.5, 0.5, 0.5, 0.5], set())
        tracker.update_round(5, [0.7, 0.5, 0.5, 0.5, 0.5], set())

        assert tracker.get_status(0) == AcceptanceStatus.PROBATIONARY

        # 3rd clean round after reset - now accepted
        tracker.update_round(6, [0.7, 0.5, 0.5, 0.5, 0.5], set())
        assert tracker.get_status(0) == AcceptanceStatus.ACCEPTED

    def test_post_acceptance_attack_transitions_to_suspicious(self):
        """Flagging an ACCEPTED node transitions to SUSPICIOUS."""
        tracker = AcceptanceTracker(
            num_nodes=5,
            acceptance_threshold=0.6,
            consecutive_clean_required=3,
            anomaly_revoke_threshold=2,
        )

        # Node 0 achieves acceptance
        for r in range(1, 4):
            tracker.update_round(r, [0.7, 0.5, 0.5, 0.5, 0.5], set())

        assert tracker.get_status(0) == AcceptanceStatus.ACCEPTED

        # Node 0 is flagged (post-acceptance attack!)
        tracker.update_round(4, [0.6, 0.5, 0.5, 0.5, 0.5], {0})

        assert tracker.get_status(0) == AcceptanceStatus.SUSPICIOUS
        assert tracker.is_post_acceptance_anomaly(0)

    def test_suspicious_to_revoked_after_multiple_flags(self):
        """Repeated flags on SUSPICIOUS node leads to REVOKED."""
        tracker = AcceptanceTracker(
            num_nodes=5,
            acceptance_threshold=0.6,
            consecutive_clean_required=3,
            anomaly_revoke_threshold=2,
        )

        # Node 0 achieves acceptance
        for r in range(1, 4):
            tracker.update_round(r, [0.7, 0.5, 0.5, 0.5, 0.5], set())

        # First flag -> SUSPICIOUS
        tracker.update_round(4, [0.6, 0.5, 0.5, 0.5, 0.5], {0})
        assert tracker.get_status(0) == AcceptanceStatus.SUSPICIOUS

        # Second flag -> REVOKED (threshold = 2)
        tracker.update_round(5, [0.5, 0.5, 0.5, 0.5, 0.5], {0})
        assert tracker.get_status(0) == AcceptanceStatus.REVOKED

    def test_suspicious_can_recover_to_accepted(self):
        """SUSPICIOUS node can recover to ACCEPTED with clean rounds."""
        tracker = AcceptanceTracker(
            num_nodes=5,
            acceptance_threshold=0.6,
            consecutive_clean_required=3,
            anomaly_revoke_threshold=3,  # Higher threshold
        )

        # Node 0 achieves acceptance
        for r in range(1, 4):
            tracker.update_round(r, [0.7, 0.5, 0.5, 0.5, 0.5], set())

        # First flag -> SUSPICIOUS
        tracker.update_round(4, [0.6, 0.5, 0.5, 0.5, 0.5], {0})
        assert tracker.get_status(0) == AcceptanceStatus.SUSPICIOUS

        # 3 clean rounds -> back to ACCEPTED
        for r in range(5, 8):
            tracker.update_round(r, [0.7, 0.5, 0.5, 0.5, 0.5], set())

        assert tracker.get_status(0) == AcceptanceStatus.ACCEPTED

    def test_acceptance_events_tracked(self):
        """Acceptance events are recorded for metrics."""
        tracker = AcceptanceTracker(
            num_nodes=5,
            acceptance_threshold=0.6,
            consecutive_clean_required=3,
        )

        # Node 0 achieves acceptance
        for r in range(1, 4):
            tracker.update_round(r, [0.7, 0.5, 0.5, 0.5, 0.5], set())

        assert len(tracker.acceptance_events) == 1
        event = tracker.acceptance_events[0]
        assert event["node_id"] == 0
        assert event["round"] == 3
        assert event["trust"] == 0.7


class TestStrategicAttacker:
    """Tests for delayed attack activation."""

    def test_attacker_inactive_before_start_round(self):
        """Attacker behaves honestly before attack_start_round."""
        from dfl.attacks.directed import DirectedAttacker

        config = SimulationConfig(
            num_nodes=10,
            attack_ratio=0.3,
            attack_start_round=10,
        )

        attacker = DirectedAttacker(config)
        attacker.set_current_round(5)

        # Before attack_start_round, is_active returns False
        for node_id in attacker.compromised_nodes:
            assert not attacker.is_active(node_id)

        # craft_malicious_update returns None (behave honestly)
        dummy_updates = [{"w": torch.ones(10)}]
        for node_id in attacker.compromised_nodes:
            result = attacker.craft_malicious_update(dummy_updates, node_id)
            assert result is None

    def test_attacker_active_at_start_round(self):
        """Attacker activates at attack_start_round."""
        from dfl.attacks.directed import DirectedAttacker

        config = SimulationConfig(
            num_nodes=10,
            attack_ratio=0.3,
            attack_start_round=10,
        )

        attacker = DirectedAttacker(config)
        attacker.set_current_round(10)

        # At attack_start_round, is_active returns True
        for node_id in attacker.compromised_nodes:
            assert attacker.is_active(node_id)

        # craft_malicious_update returns actual attack
        dummy_updates = [{"w": torch.ones(10)}]
        for node_id in attacker.compromised_nodes:
            result = attacker.craft_malicious_update(dummy_updates, node_id)
            assert result is not None
            assert "w" in result

    def test_attacker_active_after_start_round(self):
        """Attacker remains active after attack_start_round."""
        from dfl.attacks.directed import DirectedAttacker

        config = SimulationConfig(
            num_nodes=10,
            attack_ratio=0.3,
            attack_start_round=10,
        )

        attacker = DirectedAttacker(config)
        attacker.set_current_round(15)

        for node_id in attacker.compromised_nodes:
            assert attacker.is_active(node_id)


class TestTrustTrajectory:
    """Tests for trust trajectory tracking."""

    def test_trust_history_recorded(self):
        """Trust changes are recorded in history."""
        config = SimulationConfig(num_nodes=5)
        trust = TrustManager(config)

        trust.set_round(1)
        trust.penalize(0)

        trust.set_round(2)
        trust.penalize(0)

        trajectory = trust.get_trust_trajectory(0)
        assert len(trajectory) >= 2

    def test_trust_volatility_computation(self):
        """Volatility reflects variance in trust scores."""
        config = SimulationConfig(num_nodes=5)
        trust = TrustManager(config)

        # Penalize multiple times to create variance
        for r in range(1, 6):
            trust.set_round(r)
            if r % 2 == 0:
                trust.penalize(0)
            else:
                trust.boost(0)

        volatility = trust.compute_trust_volatility(0)
        assert volatility > 0

    def test_trust_trend_negative_when_declining(self):
        """Trend is negative when trust is declining."""
        config = SimulationConfig(num_nodes=5)
        trust = TrustManager(config)

        # Repeatedly penalize
        for r in range(1, 5):
            trust.set_round(r)
            trust.penalize(0)

        trend = trust.compute_trust_trend(0)
        assert trend < 0

    def test_peak_trust_tracking(self):
        """Peak trust correctly tracks maximum."""
        config = SimulationConfig(num_nodes=5)
        trust = TrustManager(config)

        # Boost to high trust
        for r in range(1, 4):
            trust.set_round(r)
            trust.boost(0)

        peak = trust.get_peak_trust(0)
        assert peak > 0.5  # Should be above initial

        # Penalize
        trust.set_round(5)
        trust.penalize(0)

        # Peak should remain the same
        assert trust.get_peak_trust(0) == peak

    def test_trust_drop_from_peak(self):
        """Trust drop correctly computed from peak."""
        config = SimulationConfig(num_nodes=5)
        trust = TrustManager(config)

        # Boost to high trust
        for r in range(1, 4):
            trust.set_round(r)
            trust.boost(0)

        peak_before = trust.get_peak_trust(0)

        # Penalize
        for r in range(5, 8):
            trust.set_round(r)
            trust.penalize(0)

        drop = trust.compute_trust_drop(0)
        expected_drop = peak_before - trust.get_trust(0)
        assert abs(drop - expected_drop) < 0.001


class TestConfigValidation:
    """Tests for config parameter validation."""

    def test_attack_start_round_positive(self):
        """attack_start_round must be >= 0."""
        assert_raises(ValueError, lambda: SimulationConfig(attack_start_round=-1))

    def test_attack_start_round_zero_is_valid(self):
        """attack_start_round = 0 is valid (always attack)."""
        config = SimulationConfig(attack_start_round=0)
        assert config.attack_start_round == 0

    def test_acceptance_threshold_default(self):
        """acceptance_trust_threshold has correct default."""
        config = SimulationConfig()
        assert config.acceptance_trust_threshold == 0.6


class TestAcceptanceSummary:
    """Tests for acceptance summary statistics."""

    def test_summary_counts_correct(self):
        """Summary correctly counts status distribution."""
        tracker = AcceptanceTracker(
            num_nodes=5,
            acceptance_threshold=0.6,
            consecutive_clean_required=2,
        )

        # Nodes 0, 1 achieve acceptance
        for r in range(1, 3):
            tracker.update_round(r, [0.7, 0.7, 0.5, 0.5, 0.5], set())

        # Node 0 becomes suspicious
        tracker.update_round(3, [0.6, 0.7, 0.5, 0.5, 0.5], {0})

        summary = tracker.get_summary()
        assert summary["accepted"] == 1  # Node 1
        assert summary["suspicious"] == 1  # Node 0
        assert summary["probationary"] == 3  # Nodes 2, 3, 4
