"""Make the repository root importable so tests can `import agentic_spine`
and `import experiments.execution_spine` regardless of the invocation directory.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
