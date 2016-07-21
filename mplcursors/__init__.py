from ._mplcursors import Cursor, cursor
from ._pick_info import Selection, compute_pick, get_ann_text


__all__ = ["Cursor", "cursor", "Selection", "compute_pick", "get_ann_text"]


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
