"""Acceptance status tracking for post-acceptance attack detection.

This module tracks when nodes achieve "accepted" status through honest behavior
and detects when they subsequently turn malicious (post-acceptance attacks).

Key concepts:
- PROBATIONARY: New node, not yet proven honest
- ACCEPTED: Node has demonstrated consistent honest behavior
- SUSPICIOUS: Previously accepted node showing anomalies (post-acceptance attack signal)
- REVOKED: Acceptance revoked due to repeated anomalies
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class AcceptanceStatus(Enum):
    #Possible acceptance states for a node.

    PROBATIONARY = "probationary"  # New node, not yet accepted
    ACCEPTED = "accepted"  # Proven honest behavior
    SUSPICIOUS = "suspicious"  # Was accepted, now showing anomalies
    REVOKED = "revoked"  # Acceptance revoked due to attacks


@dataclass
class AcceptanceRecord:
    #Tracks a node's acceptance history.

    node_id: int
    status: AcceptanceStatus = AcceptanceStatus.PROBATIONARY
    acceptance_round: Optional[int] = None  # Round when accepted
    consecutive_clean_rounds: int = 0  # Consecutive rounds without flags
    trust_at_acceptance: float = 0.5  # Trust score when accepted
    anomaly_after_acceptance: int = 0  # Anomalies since acceptance
    last_anomaly_round: Optional[int] = None
    peak_trust: float = 0.5  # Maximum trust achieved


class AcceptanceTracker:
    """Tracks node acceptance status for post-acceptance attack detection.

    A node achieves ACCEPTED status when:
    - Trust score >= acceptance_threshold
    - Has consecutive_clean_required rounds without being flagged

    When an ACCEPTED node is flagged:
    - Status transitions to SUSPICIOUS (key post-acceptance attack signal)
    - This is more significant than a PROBATIONARY node being flagged

    Repeated anomalies lead to REVOKED status.
    """

    def __init__(
        self,
        num_nodes: int,
        acceptance_threshold: float = 0.6,
        consecutive_clean_required: int = 5,
        anomaly_revoke_threshold: int = 2,
    ):
        """Initialize the acceptance tracker.

        Args:
            num_nodes: Number of nodes in the network
            acceptance_threshold: Trust score required for acceptance
            consecutive_clean_required: Clean rounds needed for acceptance
            anomaly_revoke_threshold: Anomalies to revoke acceptance
        """
        self.num_nodes = num_nodes
        self.acceptance_threshold = acceptance_threshold
        self.consecutive_clean_required = consecutive_clean_required
        self.anomaly_revoke_threshold = anomaly_revoke_threshold

        # Initialize all nodes as probationary
        self.records: Dict[int, AcceptanceRecord] = {
            i: AcceptanceRecord(node_id=i) for i in range(num_nodes)
        }

        # History for metrics
        self.acceptance_events: List[dict] = []
        self.revocation_events: List[dict] = []
        self.suspicious_events: List[dict] = []

    def update_round(
        self,
        round_num: int,
        trust_scores: List[float],
        flagged_nodes: Set[int],
    ) -> Dict[int, AcceptanceStatus]:
        """Update acceptance status based on round results.

        Args:
            round_num: Current round number
            trust_scores: Current trust scores for all nodes
            flagged_nodes: Set of node IDs flagged this round

        Returns:
            Dict of status changes this round (node_id -> new status)
        """
        changes: Dict[int, AcceptanceStatus] = {}

        for node_id in range(self.num_nodes):
            record = self.records[node_id]
            trust = trust_scores[node_id]
            was_flagged = node_id in flagged_nodes
            old_status = record.status

            # Update peak trust
            if trust > record.peak_trust:
                record.peak_trust = trust

            if was_flagged:
                self._handle_flag(record, round_num, trust)
            else:
                self._handle_clean_round(record, round_num, trust)

            if record.status != old_status:
                changes[node_id] = record.status

        return changes

    def _handle_flag(
        self,
        record: AcceptanceRecord,
        round_num: int,
        trust: float,
    ) -> None:
        #Handle a node being flagged this round.
        record.consecutive_clean_rounds = 0
        record.last_anomaly_round = round_num

        if record.status == AcceptanceStatus.ACCEPTED:
            # POST-ACCEPTANCE ANOMALY - detection signal!
            record.status = AcceptanceStatus.SUSPICIOUS
            record.anomaly_after_acceptance += 1
            self.suspicious_events.append(
                {
                    "node_id": record.node_id,
                    "round": round_num,
                    "trust": trust,
                    "rounds_since_acceptance": round_num - (record.acceptance_round or 0),
                    "trust_drop": record.trust_at_acceptance - trust,
                }
            )

        elif record.status == AcceptanceStatus.SUSPICIOUS:
            record.anomaly_after_acceptance += 1
            if record.anomaly_after_acceptance >= self.anomaly_revoke_threshold:
                record.status = AcceptanceStatus.REVOKED
                self.revocation_events.append(
                    {
                        "node_id": record.node_id,
                        "round": round_num,
                        "anomalies_after_acceptance": record.anomaly_after_acceptance,
                        "rounds_since_acceptance": round_num
                        - (record.acceptance_round or 0),
                    }
                )

    def _handle_clean_round(
        self,
        record: AcceptanceRecord,
        round_num: int,
        trust: float,
    ) -> None:
        """Handle a node NOT being flagged this round."""
        record.consecutive_clean_rounds += 1

        if record.status == AcceptanceStatus.PROBATIONARY:
            # Check for acceptance
            if (
                trust >= self.acceptance_threshold
                and record.consecutive_clean_rounds >= self.consecutive_clean_required
            ):
                record.status = AcceptanceStatus.ACCEPTED
                record.acceptance_round = round_num
                record.trust_at_acceptance = trust
                self.acceptance_events.append(
                    {
                        "node_id": record.node_id,
                        "round": round_num,
                        "trust": trust,
                        "consecutive_clean": record.consecutive_clean_rounds,
                    }
                )

        elif record.status == AcceptanceStatus.SUSPICIOUS:
            # Recovery: enough clean rounds returns to accepted
            if record.consecutive_clean_rounds >= 3:
                record.status = AcceptanceStatus.ACCEPTED

    def is_accepted(self, node_id: int) -> bool:
        """Check if node currently has accepted status."""
        return self.records[node_id].status == AcceptanceStatus.ACCEPTED

    def is_post_acceptance_anomaly(self, node_id: int) -> bool:
        """Check if node is showing post-acceptance attack signature."""
        status = self.records[node_id].status
        return status in (AcceptanceStatus.SUSPICIOUS, AcceptanceStatus.REVOKED)

    def was_ever_accepted(self, node_id: int) -> bool:
        """Check if node was ever accepted (even if now suspicious/revoked)."""
        return self.records[node_id].acceptance_round is not None

    def get_status(self, node_id: int) -> AcceptanceStatus:
        """Get current acceptance status."""
        return self.records[node_id].status

    def get_record(self, node_id: int) -> AcceptanceRecord:
        """Get full acceptance record for a node."""
        return self.records[node_id]

    def get_accepted_nodes(self) -> Set[int]:
        """Get set of currently accepted node IDs."""
        return {
            node_id
            for node_id, record in self.records.items()
            if record.status == AcceptanceStatus.ACCEPTED
        }

    def get_suspicious_nodes(self) -> Set[int]:
        """Get set of suspicious node IDs (post-acceptance anomalies)."""
        return {
            node_id
            for node_id, record in self.records.items()
            if record.status == AcceptanceStatus.SUSPICIOUS
        }

    def get_summary(self) -> Dict:
        """Get summary of acceptance status across all nodes."""
        status_counts = {status: 0 for status in AcceptanceStatus}
        for record in self.records.values():
            status_counts[record.status] += 1

        return {
            "probationary": status_counts[AcceptanceStatus.PROBATIONARY],
            "accepted": status_counts[AcceptanceStatus.ACCEPTED],
            "suspicious": status_counts[AcceptanceStatus.SUSPICIOUS],
            "revoked": status_counts[AcceptanceStatus.REVOKED],
            "total_acceptance_events": len(self.acceptance_events),
            "total_suspicious_events": len(self.suspicious_events),
            "total_revocation_events": len(self.revocation_events),
        }
