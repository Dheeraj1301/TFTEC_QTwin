<<<<<<< HEAD
"""Expose model modules from the project subpackage."""
from pathlib import Path as _Path

_pkg = _Path(__file__).resolve().parent.parent / "ThermoTwinAI-Quantum" / "models"
__path__ = [str(_pkg)]
=======
"""ThermoTwinAI-Quantum model package.

The presence of this file marks the ``models`` directory as a package so that
modules like :mod:`models.quantum_lstm` can be imported when executing
``main.py`` directly.
"""

from .quantum_lstm import train_quantum_lstm
from .quantum_prophet import train_quantum_prophet

__all__ = ["train_quantum_lstm", "train_quantum_prophet"]
