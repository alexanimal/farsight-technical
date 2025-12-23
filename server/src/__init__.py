"""Farsight technical server package.

Note: This module does not import config at module level to avoid
non-deterministic operations (like Path.expanduser) that would break
Temporal workflow determinism. Import config directly where needed:
    from src.config import Settings, settings
"""

# Do not import config here - it uses Path.expanduser() which breaks
# Temporal workflow determinism. Import it directly where needed instead.

__all__ = []
