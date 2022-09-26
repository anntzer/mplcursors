from setupext import find_namespace_packages, setup


setup.register_pth_hook("setup_mplcursors_pth.py", "mplcursors.pth")


setup(
    name="mplcursors",
    description="Interactive data selection cursors for Matplotlib.",
    long_description=open("README.rst", encoding="utf-8").read(),
    long_description_content_type="text/x-rst",
    author="Antony Lee",
    url="https://github.com/anntzer/mplcursors",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Matplotlib",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_namespace_packages("lib"),
    package_dir={"": "lib"},
    python_requires=">=3.6",
    setup_requires=["setuptools_scm"],
    use_scm_version=lambda: {  # xref __init__.py
        "version_scheme": "post-release",
        "local_scheme": "node-and-date",
        "write_to": "lib/mplcursors/_version.py",
    },
    install_requires=[
        "matplotlib>=3.1",
    ],
    extras_require={
        "docs": [
            "pandas",
            "pydata_sphinx_theme!=0.10.1",
            "sphinx",
            "sphinx-gallery",
        ],
    },
)
