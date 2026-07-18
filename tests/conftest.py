import os
import sys

# Make config and detector importable when pytest runs from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
