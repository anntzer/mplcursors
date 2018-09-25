from setupext import find_packages, setup


# We cannot directly import matplotlib if `MPLCURSORS` is set because
# `sys.path` is not correctly set yet.
#
# The loading of `matplotlib.figure` does not go through the path entry finder
# because it is a submodule, so we must use a metapath finder instead.

@setup.register_pth_hook("mplcursors.pth")
def _pth_hook():
    if os.environ.get("MPLCURSORS"):
        from importlib.machinery import PathFinder
        class MplcursorsMetaPathFinder(PathFinder):
            def find_spec(self, fullname, path=None, target=None):
                spec = super().find_spec(fullname, path, target)
                if fullname == "matplotlib.figure":
                    def exec_module(module):
                        type(spec.loader).exec_module(spec.loader, module)
                        # The pth file does not get properly uninstalled from
                        # a develop install.  See pypa/pip#4176.
                        try:
                            import mplcursors
                        except ImportError:
                            return
                        import functools, json, weakref
                        # Ensure that when the cursor is removed(), or gets
                        # GC'd because its referents artists are GC'd, the
                        # entry also disappears.
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
                                    # No artist yet; skip possible
                                    # initialization code.
                                    cursor.remove()
                            return rv
                        module.Figure.draw = wrapper
                    spec.loader.exec_module = exec_module
                    sys.meta_path.remove(self)
                return spec
        sys.meta_path.insert(0, MplcursorsMetaPathFinder())


setup(
    name="mplcursors",
    description="Interactive data selection cursors for Matplotlib.",
    long_description=open("README.rst", encoding="utf-8").read(),
    author="Antony Lee",
    url="https://github.com/anntzer/mplcursors",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages("lib"),
    package_dir={"": "lib"},
    python_requires=">=3.4",
    setup_requires=["setuptools_scm"],
    use_scm_version=lambda: {  # xref __init__.py
        "version_scheme": "post-release",
        "local_scheme": "node-and-date",
        "write_to": "lib/mplcursors/_version.py",
    },
    install_requires=[
        "matplotlib>=2.1",
    ],
)
