import gc
import weakref

from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import mplcursors
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest


_eps = .001


@pytest.yield_fixture
def ax():
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    try:
        yield ax
    finally:
        plt.close(fig)


def _process_event(name, ax, coords, *args):
    if name == "__mouse_click__":
        # So that the dragging callbacks don't go crazy.
        _process_event("button_press_event", ax, coords, *args)
        _process_event("button_release_event", ax, coords, *args)
        return
    display_coords = ax.transData.transform_point(coords)
    if name in ["button_press_event", "button_release_event",
                "motion_notify_event", "scroll_event"]:
        event = MouseEvent(name, ax.figure.canvas, *display_coords, *args)
    elif name in ["key_press_event", "key_release_event"]:
        event = KeyEvent(name, ax.figure.canvas, *args, *display_coords)
    else:
        raise ValueError("Unknown event name {!r}".format(name))
    ax.figure.canvas.callbacks.process(name, event)


def _get_remove_args(sel):
    ax = sel.artist.axes
    # Text bounds are found only upon drawing.
    ax.figure.canvas.draw()
    bbox = sel.annotation.get_window_extent()
    center = ax.transData.inverted().transform_point(
        ((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2))
    return "__mouse_click__", ax, center, 3


def test_line(ax):
    ax.plot([0, .2, 1], [0, .8, 1], label="foo")
    cursor = mplcursors.cursor(multiple=True)
    # Far, far away.
    _process_event("__mouse_click__", ax, (0, 1), 1)
    assert len(cursor.selections) == len(ax.texts) == 0
    # On the line.
    _process_event("__mouse_click__", ax, (.1, .4), 1)
    assert len(cursor.selections) == len(ax.texts) == 1
    assert cursor.selections[0].annotation.get_text() == "foo\nx=0.1\ny=0.4"
    # Not removing it.
    _process_event("__mouse_click__", ax, (0, 1), 3)
    assert len(cursor.selections) == len(ax.texts) == 1
    # Add another one.
    _process_event("__mouse_click__", ax, (.6, .9), 1)
    assert len(cursor.selections) == len(ax.texts) == 2
    # Remove both of them (first removing the second one, to test
    # `Selection.__eq__` -- otherwise it is bypassed as `list.remove`
    # checks identity first).
    _process_event(*_get_remove_args(cursor.selections[1]))
    assert len(cursor.selections) == len(ax.texts) == 1
    _process_event(*_get_remove_args(cursor.selections[0]))
    assert len(cursor.selections) == len(ax.texts) == 0
    # Will project on the vertex at (.2, .8).
    _process_event("__mouse_click__", ax, (.2 - _eps, .8 + _eps), 1)
    assert len(cursor.selections) == len(ax.texts) == 1


@pytest.mark.parametrize(
    "plotter",
    [lambda ax, data: ax.plot(*data, "o"),
     lambda ax, data: ax.scatter(*data)])
def test_scatter(ax, plotter):
    plotter(ax, [[0, .5, 1], [0, .5, 1]])
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (.2, .2), 1)
    assert len(cursor.selections) == len(ax.texts) == 0
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.texts) == 1


def test_steps_index():
    from mplcursors._pick_info import Index
    index = Index(0, .5, .5)
    assert np.floor(index) == 0 and np.ceil(index) == 1
    assert str(index) == "0.(x=0.5, y=0.5)"


def test_steps_pre(ax):
    ax.plot([0, 1], [0, 1], drawstyle="steps-pre")
    ax.set(xlim=(-1, 2), ylim=(-1, 2))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (1, 0), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (0, .5), 1)
    index = cursor.selections[0].target.index
    assert_allclose((index.int, index.x, index.y), (0, 0, .5))
    _process_event("__mouse_click__", ax, (.5, 1), 1)
    index = cursor.selections[0].target.index
    assert_allclose((index.int, index.x, index.y), (0, .5, 1))


def test_steps_mid(ax):
    ax.plot([0, 1], [0, 1], drawstyle="steps-mid")
    ax.set(xlim=(-1, 2), ylim=(-1, 2))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (0, 1), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (1, 0), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (.25, 0), 1)
    index = cursor.selections[0].target.index
    assert_allclose((index.int, index.x, index.y), (0, .25, 0))
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    index = cursor.selections[0].target.index
    assert_allclose((index.int, index.x, index.y), (0, .5, .5))
    _process_event("__mouse_click__", ax, (.75, 1), 1)
    index = cursor.selections[0].target.index
    assert_allclose((index.int, index.x, index.y), (0, .75, 1))


def test_steps_post(ax):
    ax.plot([0, 1], [0, 1], drawstyle="steps-post")
    ax.set(xlim=(-1, 2), ylim=(-1, 2))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (0, 1), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (.5, 0), 1)
    index = cursor.selections[0].target.index
    assert_allclose((index.int, index.x, index.y), (0, .5, 0))
    _process_event("__mouse_click__", ax, (1, .5), 1)
    index = cursor.selections[0].target.index
    assert_allclose((index.int, index.x, index.y), (0, 1, .5))


def test_line_single_point(ax):
    for ls in ["-", "o"]:
        ax.cla()
        ax.plot(0, ls)
        ax.set(xlim=(-1, 1), ylim=(-1, 1))
        cursor = mplcursors.cursor()
        _process_event("__mouse_click__", ax, (_eps, _eps), 1)
        assert len(cursor.selections) == len(ax.texts) == (ls == "o")
        if cursor.selections:
            assert_array_equal(np.asarray(cursor.selections[0].target), (0, 0))
        cursor.remove()


def test_nan(ax):
    ax.plot([0, 1, np.nan, 3, 4])
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.texts) == 1
    assert_allclose(np.asarray(cursor.selections[0].target), (.5, .5))


def test_image(ax):
    ax.imshow(np.arange(4).reshape((2, 2)))
    cursor = mplcursors.cursor()
    # Not picking out-of-axes or of image.
    _process_event("__mouse_click__", ax, (-1, -1), 1)
    assert len(cursor.selections) == 0
    ax.set(xlim=(-.5, 2.5), ylim=(-.5, 2.5))
    _process_event("__mouse_click__", ax, (2, 2), 1)
    assert len(cursor.selections) == 0
    # Annotation text includes image value.
    _process_event("__mouse_click__", ax, (.75, .75), 1)
    assert (cursor.selections[0].annotation.get_text()
            == "x=0.75\ny=0.75\n[3]")
    # Moving around.
    _process_event("key_press_event", ax, (.123, .456), "shift+left")
    assert (cursor.selections[0].annotation.get_text()
            == "x=0\ny=1\n[2]")
    _process_event("key_press_event", ax, (.123, .456), "shift+right")
    assert (cursor.selections[0].annotation.get_text()
            == "x=1\ny=1\n[3]")
    _process_event("key_press_event", ax, (.123, .456), "shift+up")
    assert (cursor.selections[0].annotation.get_text()
            == "x=1\ny=0\n[1]")
    _process_event("key_press_event", ax, (.123, .456), "shift+down")
    assert (cursor.selections[0].annotation.get_text()
            == "x=1\ny=1\n[3]")


def test_container(ax):
    ax.bar(range(3), [1] * 3)
    assert len(mplcursors.cursor().artists) == 3


def test_misc_artists(ax):
    # Texts should not trigger a warning.
    text = ax.text(.5, .5, "foo")
    cursor = mplcursors.cursor(text)
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    ax.cla()
    # Other unsupported artists should.
    coll = ax.fill_between([0, 1], 0, 1)
    cursor = mplcursors.cursor(coll)
    with pytest.warns(UserWarning):
        _process_event("__mouse_click__", ax, (.5, .5), 1)


def test_cropped_by_axes():
    fig, axs = plt.subplots(2)
    axs[0].plot([0, 0], [0, 1])
    # Pan to hide the line behind the second axes.
    axs[0].set(xlim=(-1, 1), ylim=(1, 2))
    axs[1].set(xlim=(-1, 1), ylim=(-1, 1))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", axs[1], (0, 0), 1)
    assert len(cursor.selections) == 0


def test_move(ax):
    ax.plot([0, 1], [0, 1])
    cursor = mplcursors.cursor()
    # Nothing happens with no cursor.
    _process_event("key_press_event", ax, (.123, .456), "shift+left")
    assert len(cursor.selections) == 0
    # Now we move the cursor left or right.
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert tuple(cursor.selections[0].target) == (.5, .5)
    _process_event("key_press_event", ax, (.123, .456), "shift+left")
    assert tuple(cursor.selections[0].target) == (0, 0)
    assert cursor.selections[0].target.index == 0
    _process_event("key_press_event", ax, (.123, .456), "shift+right")
    assert tuple(cursor.selections[0].target) == (1, 1)
    assert cursor.selections[0].target.index == 1


def test_hover(ax):
    l1, = ax.plot([0, 1])
    l2, = ax.plot([1, 2])
    cursor = mplcursors.cursor(hover=True)
    _process_event("motion_notify_event", ax, (.5, .5))
    assert cursor.selections[0].artist == l1
    _process_event("motion_notify_event", ax, (.5, 1.5))
    assert cursor.selections[0].artist == l2


def test_highlight(ax):
    ax.plot([0, 1])
    cursor = mplcursors.cursor(highlight=True)
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert ax.artists == cursor.selections[0].extras != []
    _process_event(*_get_remove_args(cursor.selections[0]))
    assert len(ax.artists) == 0


def test_callback(ax):
    ax.plot([0, 1])
    calls = []
    cursor = mplcursors.cursor()
    @cursor.connect("add")
    def on_add(sel):
        calls.append(sel)
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(calls) == 1
    cursor.disconnect(on_add)
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(calls) == 1


def test_removed_artist(ax):
    l, = ax.plot([0, 1])
    cursor = mplcursors.cursor()
    l.remove()
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.texts) == 0


def test_remove(ax):
    ax.plot([0, 1])
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.texts) == 1
    cursor.remove()
    assert len(cursor.selections) == len(ax.texts) == 0
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.texts) == 0


def test_keys(ax):
    ax.plot([0, 1])
    cursor = mplcursors.cursor(multiple=True)
    _process_event("__mouse_click__", ax, (.3, .3), 1)
    # Toggle visibility.
    _process_event("key_press_event", ax, (.123, .456), "d")
    assert not cursor.selections[0].annotation.get_visible()
    _process_event("key_press_event", ax, (.123, .456), "d")
    assert cursor.selections[0].annotation.get_visible()
    # Disable the cursor.
    _process_event("key_press_event", ax, (.123, .456), "t")
    assert not cursor.enabled
    # (Adding becomes inactive.)
    _process_event("__mouse_click__", ax, (.6, .6), 1)
    assert len(cursor.selections) == 1
    # (Removing becomes inactive.)
    ax.figure.canvas.draw()
    _process_event(*_get_remove_args(cursor.selections[0]))
    assert len(cursor.selections) == 1
    # Reenable it.
    _process_event("key_press_event", ax, (.123, .456), "t")
    assert cursor.enabled
    _process_event(*_get_remove_args(cursor.selections[0]))
    assert len(cursor.selections) == 0


def test_convenience(ax):
    l, = ax.plot([1, 2])
    cursor = mplcursors.cursor()
    assert len(cursor.artists) == 1
    cursor = mplcursors.cursor(ax)
    assert len(cursor.artists) == 1
    cursor = mplcursors.cursor(l)
    assert len(cursor.artists) == 1
    cursor = mplcursors.cursor([l])
    assert len(cursor.artists) == 1


def test_invalid_args():
    pytest.raises(ValueError, mplcursors.cursor,
                  multiple=True, hover=True)
    pytest.raises(ValueError, mplcursors.cursor,
                  bindings={"foo": 42})
    pytest.raises(ValueError, mplcursors.cursor,
                  bindings={"select": 1, "deselect": 1})
    pytest.raises(ValueError, mplcursors.cursor().connect,
                  "foo")


def test_multiple_figures(ax):
    ax1 = ax
    fig, ax2 = plt.subplots()
    ax1.plot([0, 1])
    ax2.plot([0, 1])
    cursor = mplcursors.cursor([ax1, ax2], multiple=True)
    # Add something on the first axes.
    _process_event("__mouse_click__", ax1, (.5, .5), 1)
    assert len(cursor.selections) == 1
    assert len(ax1.texts) == 1
    assert len(ax2.texts) == 0
    # Right-clicking on the second axis doesn't remove it.
    remove_args = list(_get_remove_args(cursor.selections[0]))
    remove_args[remove_args.index(ax1)] = ax2
    _process_event(*remove_args)
    assert len(cursor.selections) == 1
    assert len(ax1.texts) == 1
    assert len(ax2.texts) == 0
    # Remove it, add something on the second.
    _process_event(*_get_remove_args(cursor.selections[0]))
    _process_event("__mouse_click__", ax2, (.5, .5), 1)
    assert len(cursor.selections) == 1
    assert len(ax1.texts) == 0
    assert len(ax2.texts) == 1


def test_gc(ax):
    def inner():
        img = ax.imshow([[0, 1], [2, 3]])
        cursor = mplcursors.cursor(img)
        f_img = weakref.finalize(img, lambda: None)
        f_cursor = weakref.finalize(cursor, lambda: None)
        img.remove()
        return f_img, f_cursor
    f_img, f_cursor = inner()
    gc.collect()
    assert not f_img.alive
    assert not f_cursor.alive
