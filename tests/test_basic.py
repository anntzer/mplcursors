from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent
import mplcursors


def _make_mouse_event(name, ax, coords, button):
    return MouseEvent(name,
                      ax.figure.canvas,
                      *ax.transData.transform_point(coords),
                      button)


def _process_mouse_event(name, ax, coords, button):
    ax.figure.canvas.callbacks.process(
        name, _make_mouse_event(name, ax, coords, button))


def test_basic():
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    cursor = mplcursors.cursor()
    _process_mouse_event("button_press_event", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.texts) == 1
