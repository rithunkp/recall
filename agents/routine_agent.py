"""Routine Agent placeholder.

Hour 3-4 will add a self-supervised hourly timestamp histogram for novelty by
time of day, trained with zero labels.
"""

import config


def config_seed() -> int:
    """Expose the configured seed for early smoke checks."""
    return config.SEED
