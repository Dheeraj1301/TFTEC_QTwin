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

__all__ = []
>>>>>>> 4ad812c43a95a4e99c2a9320230fd529e98beae6
