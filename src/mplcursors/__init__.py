import json
import os

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
           "compute_pick", "get_ann_text", "move", "make_highlight", "install"]


def install(figure):
    """
    A hook function that can be registered into ``rcParams["figure.hooks"]``.

    This hook arranges for a cursor to be registered on each figure the first
    time it is drawn, if the :envvar:`MPLCURSORS` environment variable is not
    empty (at first-draw time).  That variable must contain a JSON-encoded dict
    of options passed to `.cursor`.
    """

    def connect(event):
        figure.canvas.mpl_disconnect(cid)
        envopt = os.environ.get("MPLCURSORS")
        if not envopt:
            return
        cursor(figure, **json.loads(envopt))

    cid = figure.canvas.mpl_connect("draw_event", connect)
