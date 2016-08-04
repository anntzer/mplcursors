from ._mplcursors import Cursor, cursor
from ._pick_info import Selection, compute_pick, get_ann_text, make_highlight


__all__ = ["Cursor", "cursor", "Selection",
           "compute_pick", "get_ann_text", "make_highlight"]


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
