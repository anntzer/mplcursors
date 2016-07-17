from . import _mplcursors, _convenience
from ._mplcursors import *
from ._convenience import *


__all__ = _mplcursors.__all__ + _convenience.__all__

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
