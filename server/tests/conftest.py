"""Pytest configuration and fixtures.

This module configures pytest to properly resolve imports from the src package.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add the server directory to Python path so src imports work
# This allows tests to import from src.* modules
server_dir = Path(__file__).parent.parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

# Mock pinecone module before any imports to avoid deprecated plugin errors
# This is needed because src.db.__init__.py imports pinecone_client which imports pinecone
# The pinecone package checks for deprecated plugins at import time
if "pinecone" not in sys.modules:
    _mock_pinecone = MagicMock()
    _mock_pinecone.Pinecone = MagicMock()
    sys.modules["pinecone"] = _mock_pinecone
    sys.modules["pinecone.deprecated_plugins"] = MagicMock()
