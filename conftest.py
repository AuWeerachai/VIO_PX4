"""Test working-tree helpers, never a stale colcon install from the shell overlay."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'vio_px4_bridge'))
