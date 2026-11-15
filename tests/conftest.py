import os
import sys

# Make config and detector importable when pytest runs from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing backend.app sets up logging; send test-run output to a throwaway
# folder so it never lands in the real logs/visionguard.log.
import tempfile  # noqa: E402

import config  # noqa: E402

config.LOG_DIR = tempfile.mkdtemp(prefix="visionguard-test-logs-")
