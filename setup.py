from collections import ChainMap
from pathlib import Path
import sys
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
# The loading of `pyplot` does not go through the path entry finder because it
# is a submodule, so we must use a metapath finder instead.

pth_src = """\
if os.environ.get("MPLCURSORS"):
    from importlib.machinery import PathFinder
    class PyplotMetaPathFinder(PathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = super().find_spec(fullname, path, target)
            if fullname == "matplotlib.pyplot":
                def exec_module(module):
                    type(spec.loader).exec_module(spec.loader, module)
                    plt = module
                    import functools, json, mplcursors
                    options = json.loads(os.environ["MPLCURSORS"])
                    @functools.wraps(plt.show)
                    def wrapper(*args, **kwargs):
                        mplcursors.cursor(**options)
                        return wrapper.__wrapped__(*args, **kwargs)
                    plt.show = wrapper
                spec.loader.exec_module = exec_module
                sys.meta_path.remove(self)
            return spec
    sys.meta_path.insert(0, PyplotMetaPathFinder())
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
      description="Interactive, clickable annotations for matplotlib",
      long_description=open("README.rst").read(),
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
