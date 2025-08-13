"""ThermoTwinAI-Quantum model package.

This file makes the ``models`` directory importable and re-exports the
training helpers defined in :mod:`quantum_lstm` and
:mod:`quantum_prophet`.  Earlier merge artifacts left conflict markers in
this module, producing a ``SyntaxError`` when ``main.py`` attempted to
import it.  The conflict markers have been removed and the public API is
now explicitly defined via ``__all__``.
"""

from .quantum_lstm import train_quantum_lstm
from .quantum_prophet import train_quantum_prophet

__all__ = ["train_quantum_lstm", "train_quantum_prophet"]
