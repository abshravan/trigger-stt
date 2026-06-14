"""Pytest configuration: make the project root importable.

This lets tests use the same absolute imports as the application
(``import config``, ``from webhook.n8n_client import ...``) regardless of the
directory pytest is invoked from.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
