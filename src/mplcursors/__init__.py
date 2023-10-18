try:
    import importlib.metadata as _im
except ImportError:
    import importlib_metadata as _im
try:
    __version__ = _im.version("mplcursors")
except ImportError:
    __version__ = "0+unknown"


from ._mplcursors import Cursor, HoverMode, cursor
from ._pick_info import (
    Selection, compute_pick, get_ann_text, move, make_highlight)


__all__ = ["Cursor", "HoverMode", "cursor", "Selection",
           "compute_pick", "get_ann_text", "move", "make_highlight"]
