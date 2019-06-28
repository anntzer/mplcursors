"""
setuptools helper.

The following decorators are provided::

    # Add a pth hook.
    @setup.register_pth_hook("hook_name.pth")
    def _hook():
        # hook contents.
"""

from distutils.version import LooseVersion
from functools import partial
import inspect
import re
from pathlib import Path

import setuptools
from setuptools import Extension, find_namespace_packages, find_packages
from setuptools.command.develop import develop
from setuptools.command.install_lib import install_lib


if LooseVersion(setuptools.__version__) < "40.1":  # find_namespace_packages
    raise ImportError("setuptools>=40.1 is required")


__all__ = ["Extension", "find_namespace_packages", "find_packages", "setup"]


_pth_hooks = []


class pth_hook_mixin:
    def run(self):
        super().run()
        for fname, name, source in _pth_hooks:
            with Path(self.install_dir, fname).open("w") as file:
                file.write("import os; exec({!r}); {}()"
                           .format(source, name))

    def get_outputs(self):
        return (super().get_outputs()
                + [str(Path(self.install_dir, fname))
                   for fname, _, _ in _pth_hooks])


def setup(**kwargs):
    cmdclass = kwargs.setdefault("cmdclass", {})
    cmdclass["develop"] = type(
        "develop_with_pth_hook",
        (pth_hook_mixin, cmdclass.get("develop", develop)),
        {})
    cmdclass["install_lib"] = type(
        "install_lib_with_pth_hook",
        (pth_hook_mixin, cmdclass.get("install_lib", install_lib)),
        {})
    setuptools.setup(**kwargs)


def register_pth_hook(fname, func=None):
    if func is None:
        return partial(register_pth_hook, fname)
    source = inspect.getsource(func)
    if not re.match(r"\A@setup\.register_pth_hook.*\ndef ", source):
        raise SyntaxError("register_pth_hook must be used as a toplevel "
                          "decorator to a function")
    _, source = source.split("\n", 1)
    d = {}
    exec(source, {}, d)
    if set(d) != {func.__name__}:
        raise SyntaxError(
            "register_pth_hook should define a single function")
    _pth_hooks.append((fname, func.__name__, source))


setup.register_pth_hook = register_pth_hook
