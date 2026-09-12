"""Frozen-embedding linear probe placeholder.

Evaluation will train only the linear probe on allowed train labels, comparing
active-sampled labels against random labels at matched budgets.
"""

import config


def config_seed() -> int:
    """Expose the configured seed for early smoke checks."""
    return config.SEED
