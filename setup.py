from setuptools import find_packages, setup
import versioneer


if __name__ == "__main__":
    setup(name="mplcursors",
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          author="Antony Lee",
          license="BSD",
          classifiers=["Development Status :: 4 - Beta",
                       "License :: OSI Approved :: BSD License",
                       "Programming Language :: Python :: 3.5"],
          packages=find_packages())
