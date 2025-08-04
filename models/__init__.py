"""Expose model modules from the project subpackage."""
from pathlib import Path as _Path

_pkg = _Path(__file__).resolve().parent.parent / "ThermoTwinAI-Quantum" / "models"
__path__ = [str(_pkg)]
