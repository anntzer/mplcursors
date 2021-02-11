try:
    from ._version import version as __version__
except ImportError:
    __version__ = "(unknown version)"


from ._mplcursors import Cursor, HoverMode, cursor
from ._pick_info import (
    Selection, compute_pick, get_ann_text, move, make_highlight)


__all__ = ["Cursor", "HoverMode", "cursor", "Selection",
           "compute_pick", "get_ann_text", "move", "make_highlight"]
