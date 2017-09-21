from collections import ChainMap
import inspect
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install_lib import install_lib
import versioneer


# Environment variable-based activation.
#
# Technique inspired from http://github.com/ionelmc/python-hunter.
#
# Patch `develop` and `install_lib` instead of `build_py` because the latter is
# already hooked by versioneer.
#
# We cannot directly import matplotlib if `MPLCURSORS` is set because
# `sys.path` is not correctly set yet.
#
# The loading of `matplotlib.figure` does not go through the path entry finder
# because it is a submodule, so we must use a metapath finder instead.

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


class _pth_command_mixin:
    def run(self):
        super().run()
        with Path(self.install_dir, "mplcursors.pth").open("w") as file:
            file.write("import os; exec({!r}); _pth_hook()"
                       .format(inspect.getsource(_pth_hook)))

    def get_outputs(self):
        return (super().get_outputs()
                + [str(Path(self.install_dir, "mplcairo.pth"))])


setup(name="mplcursors",
      description="Interactive data selection cursors for Matplotlib.",
      long_description=open("README.rst", encoding="utf-8").read(),
      version=versioneer.get_version(),
      cmdclass=ChainMap(
          versioneer.get_cmdclass(),
          {"develop": type("", (_pth_command_mixin, develop), {}),
           "install_lib": type("", (_pth_command_mixin, install_lib), {})}),
      author="Antony Lee",
      url="https://github.com/anntzer/mplcursors",
      license="MIT",
      classifiers=[
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: MIT License",
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
