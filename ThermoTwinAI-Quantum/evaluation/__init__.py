<<<<<<< HEAD
"""Expose evaluation modules from the project subpackage."""
from pathlib import Path as _Path

_pkg = _Path(__file__).resolve().parent.parent / "ThermoTwinAI-Quantum" / "evaluation"
__path__ = [str(_pkg)]
=======
"""Evaluation utilities for model assessment."""

from .evaluate_models import evaluate_model

__all__ = ["evaluate_model"]
>>>>>>> 9ab55c7a0fdef0a8a6e4778f033b7ca209c94343
