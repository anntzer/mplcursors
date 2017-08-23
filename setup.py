from collections import ChainMap
from pathlib import Path
from tempfile import NamedTemporaryFile

from setuptools import find_packages, setup
from setuptools.command.install_lib import install_lib
import versioneer


# Environment variable-based activation.
#
# Technique inspired from http://github.com/ionelmc/python-hunter.
#
# Patch `install_lib` instead of `build_py` because the latter is already
# hooked by versioneer.
#
# We cannot directly import matplotlib if `MPLCURSORS` is set because
# `sys.path` is not correctly set yet.
#
# The loading of `matplotlib.figure` does not go through the path entry finder
# because it is a submodule, so we must use a metapath finder instead.

pth_src = """\
if os.environ.get("MPLCURSORS"):
    from importlib.machinery import PathFinder
    class MplcursorsMetaPathFinder(PathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = super().find_spec(fullname, path, target)
            if fullname == "matplotlib.figure":
                def exec_module(module):
                    type(spec.loader).exec_module(spec.loader, module)
                    import functools, json, weakref, mplcursors
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
                                # No artist yet; skip possible initialization
                                # code.
                                cursor.remove()
                        return rv
                    module.Figure.draw = wrapper
                spec.loader.exec_module = exec_module
                sys.meta_path.remove(self)
            return spec
    sys.meta_path.insert(0, MplcursorsMetaPathFinder())
"""


class install_lib_with_pth(install_lib):
    def run(self):
        super().run()
        with NamedTemporaryFile("w") as file:
            file.write("import os; exec({!r})".format(pth_src))
            file.flush()
            self.copy_file(
                file.name, str(Path(self.install_dir, "mplcursors.pth")))


setup(name="mplcursors",
      description="Interactive data selection cursors for Matplotlib.",
      long_description=open("README.rst", encoding="utf-8").read(),
      version=versioneer.get_version(),
      cmdclass=ChainMap(versioneer.get_cmdclass(),
                        {"install_lib": install_lib_with_pth}),
      author="Antony Lee",
      url="https://github.com/anntzer/mplcursors",
      license="BSD",
      classifiers=[
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: BSD License",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6"
      ],
      packages=find_packages(include=["mplcursors", "mplcursors.*"]),
      python_requires=">=3.4",
      install_requires=[
          "matplotlib>=2.0"
      ],
      )
