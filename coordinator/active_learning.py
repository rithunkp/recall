"""Active-learning placeholder.

Coordinator disagreements will become uncertainty queries. Labels gathered here
may update train-time prototypes only; the test split remains untouched until
final evaluation.
"""

import config


def config_seed() -> int:
    """Expose the configured seed for early smoke checks."""
    return config.SEED
