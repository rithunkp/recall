"""Memory storage placeholder.

Stored memories will include embeddings, thumbnails, timestamps, and metadata
under the configured memory artifact directories.
"""

import config


def memory_root() -> str:
    """Return the configured memory artifact directory."""
    return str(config.MEMORY_DIR)
