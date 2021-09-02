"""setuptools helpers."""

import functools
import inspect
from pathlib import Path
import re

import setuptools
# find_namespace_packages itself bounds support to setuptools>=40.1.
from setuptools import Distribution, Extension, find_namespace_packages


__all__ = ["Extension", "find_namespace_packages", "setup"]


def register_pth_hook(fname, func=None):
    """
    ::
        # Add a pth hook.
        @setup.register_pth_hook("hook_name.pth")
        def _hook():
            '''hook contents.'''
    """
    if func is None:
        return functools.partial(register_pth_hook, fname)
    source = inspect.getsource(func)
    if not re.match(
            rf"@setup\.register_pth_hook.*\ndef {re.escape(func.__name__)}\(",
            source):
        raise SyntaxError("register_pth_hook must be used as a toplevel "
                          "decorator to a function")
    _, source = source.split("\n", 1)
    _pth_hook_mixin._pth_hooks.append((fname, func.__name__, source))


class _pth_hook_mixin:
    _pth_hooks = []

    def run(self):
        super().run()
        for fname, name, source in self._pth_hooks:
            with Path(self.install_dir, fname).open("w") as file:
                file.write(f"import os; exec({source!r}); {name}()")

    def get_outputs(self):
        return (super().get_outputs()
                + [str(Path(self.install_dir, fname))
                   for fname, _, _ in self._pth_hooks])


def _prepare_pth_hook(kwargs):
    cmdclass = kwargs.setdefault("cmdclass", {})
    get = Distribution({"cmdclass": cmdclass}).get_command_class
    cmdclass["develop"] = type(
        "develop_with_pth_hook", (_pth_hook_mixin, get("develop")), {})
    cmdclass["install_lib"] = type(
        "install_lib_with_pth_hook", (_pth_hook_mixin, get("install_lib")), {})


def setup(**kwargs):
    _prepare_pth_hook(kwargs)
    setuptools.setup(**kwargs)


setup.register_pth_hook = register_pth_hook
