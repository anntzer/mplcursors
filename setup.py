import sys

from setuptools import find_packages, setup
import versioneer


if __name__ == "__main__":
    if not sys.version_info >= (3, 5):
        raise ImportError("mplcursors require Python>=3.5")

    setup(name="mplcursors",
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          author="Antony Lee",
          license="BSD",
          classifiers=["Development Status :: 4 - Beta",
                       "License :: OSI Approved :: BSD License",
                       "Programming Language :: Python :: 3.5"],
          packages=find_packages(),
          install_requires=["matplotlib>=1.5.0"])
