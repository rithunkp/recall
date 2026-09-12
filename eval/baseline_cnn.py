"""Mandatory scratch-CNN baseline placeholder.

This is the supervised baseline trained from scratch on the identical random
1%/10% labeled train subset used by the comparison protocol.
"""

import config


def config_seed() -> int:
    """Expose the configured seed for early smoke checks."""
    return config.SEED
