from pathlib import Path
import tokenize

import setuptools
from setuptools import Distribution


def register_pth_hook(source_path, pth_name):
    """
    ::
        setup.register_pth_hook("hook_source.py", "hook_name.pth")  # Add hook.
    """
    with tokenize.open(source_path) as file:
        source = file.read()
    _pth_hook_mixin._pth_hooks.append((pth_name, source))


class _pth_hook_mixin:
    _pth_hooks = []

    def run(self):
        super().run()
        for pth_name, source in self._pth_hooks:
            with Path(self.install_dir, pth_name).open("w") as file:
                file.write(f"import os; exec({source!r})")

    def get_outputs(self):
        return (super().get_outputs()
                + [str(Path(self.install_dir, pth_name))
                   for pth_name, _ in self._pth_hooks])


def setup(**kwargs):
    cmdclass = kwargs.setdefault("cmdclass", {})
    get = Distribution({"cmdclass": cmdclass}).get_command_class
    cmdclass["develop"] = type(
        "develop_with_pth_hook", (_pth_hook_mixin, get("develop")), {})
    cmdclass["install_lib"] = type(
        "install_lib_with_pth_hook", (_pth_hook_mixin, get("install_lib")), {})
    setuptools.setup(**kwargs)


register_pth_hook("setup_mplcursors_pth.py", "mplcursors.pth")
setup()
