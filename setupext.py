from functools import partial
import inspect
import re
from pathlib import Path

import setuptools
from setuptools import Extension, find_packages
from setuptools.command.develop import develop
from setuptools.command.install_lib import install_lib


__all__ = ["setup", "get_pybind_include",
           "Extension", "find_packages"]


class setup:
    _pth_hooks = {}

    def __new__(cls, **kwargs):

        cmdclass = kwargs.setdefault("cmdclass", {})

        class pth_hook_mixin:

            def run(self):
                super().run()
                for fname, (name, source) in cls._pth_hooks.items():
                    with Path(self.install_dir, fname).open("w") as file:
                        file.write("import os; exec({!r}); {}()"
                                   .format(source, name))

            def get_outputs(self):
                return (super().get_outputs()
                        + [str(Path(self.install_dir, fname))
                           for fname in cls._pth_hooks])

        cmdclass["develop"] = type(
            "develop_with_pth_hook",
            (pth_hook_mixin, cmdclass.get("develop", develop)),
            {})
        cmdclass["install_lib"] = type(
            "install_lib_with_pth_hook",
            (pth_hook_mixin, cmdclass.get("install_lib", install_lib)),
            {})

        setuptools.setup(**kwargs)

    @classmethod
    def register_pth_hook(cls, fname, func=None):
        if func is None:
            return partial(cls.register_pth_hook, fname)
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
        cls._pth_hooks[fname] = func.__name__, source


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path.

    The purpose of this class is to postpone importing pybind11 until it is
    actually installed, so that the ``get_include()`` method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)
