"""Memory retrieval placeholder.

Hour 5-6 will add CLIP-text nearest-neighbor retrieval over stored memory
embeddings without generation.
"""

import config


def config_seed() -> int:
    """Expose the configured seed for early smoke checks."""
    return config.SEED
