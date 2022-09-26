import os


if os.environ.get("MPLCURSORS"):
    # We cannot directly import matplotlib if `MPLCURSORS` is set because
    # `sys.path` is not correctly set yet.
    # The loading of `matplotlib.figure` does not go through the path entry
    # finder because it is a submodule, so we use a metapath finder instead.

    from importlib.machinery import PathFinder
    import sys

    class MplcursorsMetaPathFinder(PathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = super().find_spec(fullname, path, target)
            if fullname == "matplotlib.figure":
                def exec_module(module):
                    type(spec.loader).exec_module(spec.loader, module)
                    # The pth file does not get properly uninstalled from a
                    # develop install.  See pypa/pip#4176.
                    try:
                        import mplcursors
                    except ImportError:
                        return
                    import functools
                    import json
                    import weakref
                    # Ensure that when the cursor is removed(), or gets GC'd
                    # because its referents artists are GC'd, the entry also
                    # disappears.
                    cursors = weakref.WeakValueDictionary()
                    options = json.loads(os.environ["MPLCURSORS"])
                    @functools.wraps(module.Figure.draw)
                    def wrapper(self, *args, **kwargs):
                        rv = wrapper.__wrapped__(self, *args, **kwargs)
                        if self not in cursors:
                            cursor = mplcursors.cursor(self, **options)
                            if cursor.artists:
                                cursors[self] = cursor
                            else:
                                # No artist yet; skip possible init code.
                                cursor.remove()
                        return rv
                    module.Figure.draw = wrapper
                spec.loader.exec_module = exec_module
                sys.meta_path.remove(self)
            return spec

    sys.meta_path.insert(0, MplcursorsMetaPathFinder())
