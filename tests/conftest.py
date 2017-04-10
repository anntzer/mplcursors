from pathlib import Path


def pytest_make_parametrize_id(config, val):
    if isinstance(val, type(lambda: None)) and val.__qualname__ != "<lambda>":
        return val.__qualname__
    if isinstance(val, Path):
        return str(val)
