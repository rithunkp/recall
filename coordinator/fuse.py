"""Coordinator fusion placeholder.

Hour 4-5 will combine agent novelty votes with the configured majority rule and
route disagreements to active learning.
"""

import config


def min_votes() -> int:
    """Return the configured vote count needed for a novel decision."""
    return config.COORDINATOR_MIN_VOTES
