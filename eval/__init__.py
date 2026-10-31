"""Independent SARIF evaluation: no dependency on analyzer implementation."""

from .metrics import EvaluationError, evaluate
from .freeze import create_freeze, verify_frozen

__all__ = ["EvaluationError", "evaluate", "create_freeze", "verify_frozen"]
