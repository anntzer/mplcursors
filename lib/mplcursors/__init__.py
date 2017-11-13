try:
    import setuptools_scm
    __version__ = setuptools_scm.get_version(  # xref setup.py
        root="../..", relative_to=__file__,
        version_scheme="post-release", local_scheme="node-and-date")
except (ImportError, LookupError):
    try:
        from ._version import version as __version__
    except ImportError:
        pass


from ._mplcursors import Cursor, cursor
from ._pick_info import Selection, compute_pick, get_ann_text, make_highlight


__all__ = ["Cursor", "cursor", "Selection",
           "compute_pick", "get_ann_text", "make_highlight"]
