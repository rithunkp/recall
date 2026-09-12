"""Semantic Agent placeholder.

Hour 2-3 will add frozen CLIP image embeddings and category-level novelty
scoring against familiar prototypes. CLIP must remain frozen.
"""

import config


def config_seed() -> int:
    """Expose the configured seed for early smoke checks."""
    return config.SEED
