import copy
import functools
import gc
import os
from pathlib import Path
import re
import subprocess
import sys
import weakref

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
import mplcursors
from mplcursors import _pick_info, Selection, HoverMode
import numpy as np
import pytest


# The absolute tolerance is quite large to take into account rounding of
# LocationEvents to the nearest pixel by Matplotlib, which causes a relative
# error of ~ 1/#pixels.
approx = functools.partial(pytest.approx, abs=1e-2)


@pytest.fixture
def fig():
    fig = plt.figure(1)
    fig.canvas.callbacks.exception_handler = None
    return fig


@pytest.fixture
def ax(fig):
    return fig.add_subplot(111)


@pytest.fixture(autouse=True)
def cleanup():
    with mpl.rc_context({"axes.unicode_minus": False}):
        try:
            yield
        finally:
            mplcursors.__warningregistry__ = {}
            plt.close("all")


def _internal_warnings(record):
    return [
        warning for warning in record
        if Path(mplcursors.__file__).parent in Path(warning.filename).parents]


def _process_event(name, ax, coords, *args):
    ax.viewLim  # unstale viewLim.
    if name == "__mouse_click__":
        # So that the dragging callbacks don't go crazy.
        _process_event("button_press_event", ax, coords, *args)
        _process_event("button_release_event", ax, coords, *args)
        return
    display_coords = ax.transData.transform(coords)
    if name in ["button_press_event", "button_release_event",
                "motion_notify_event", "scroll_event"]:
        event = MouseEvent(name, ax.figure.canvas, *display_coords, *args)
    elif name in ["key_press_event", "key_release_event"]:
        event = KeyEvent(name, ax.figure.canvas, *args, *display_coords)
    else:
        raise ValueError(f"Unknown event name {name!r}")
    ax.figure.canvas.callbacks.process(name, event)


def _get_remove_args(sel):
    ax = sel.artist.axes
    # Text bounds are found only upon drawing.
    ax.figure.canvas.draw()
    bbox = sel.annotation.get_window_extent()
    center = ax.transData.inverted().transform(
        ((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2))
    return "__mouse_click__", ax, center, 3


def _parse_annotation(sel, regex):
    result = re.fullmatch(regex, sel.annotation.get_text())
    assert result, \
        "{!r} doesn't match {!r}".format(sel.annotation.get_text(), regex)
    return tuple(map(float, result.groups()))


def test_containerartist(ax):
    artist = _pick_info.ContainerArtist(ax.errorbar([], []))
    str(artist)
    repr(artist)


def test_selection_identity_comparison():
    sel0, sel1 = [Selection(artist=None,
                            target_=np.array([0, 0]),
                            index=None,
                            dist=0,
                            annotation=None,
                            extras=[])
                  for _ in range(2)]
    assert sel0 != sel1


def test_degenerate_inputs(ax):
    empty_container = ax.bar([], [])
    assert not mplcursors.cursor().artists
    assert not mplcursors.cursor(empty_container).artists
    pytest.raises(TypeError, mplcursors.cursor, [1])


@pytest.mark.parametrize("plotter", [Axes.plot, Axes.fill])
def test_line(ax, plotter):
    artist, = plotter(ax, [0, .2, 1], [0, .8, 1], label="foo")
    cursor = mplcursors.cursor(multiple=True)
    # Far, far away.
    _process_event("__mouse_click__", ax, (0, 1), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == 0
    # On the line.
    _process_event("__mouse_click__", ax, (.1, .4), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == 1
    assert _parse_annotation(
        cursor.selections[0], "foo\nx=(.*)\ny=(.*)") == approx((.1, .4))
    # Not removing it.
    _process_event("__mouse_click__", ax, (0, 1), 3)
    assert len(cursor.selections) == len(ax.figure.artists) == 1
    # Remove the text label; add another annotation.
    artist.set_label(None)
    _process_event("__mouse_click__", ax, (.6, .9), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == 2
    assert _parse_annotation(
        cursor.selections[1], "x=(.*)\ny=(.*)") == approx((.6, .9))
    # Remove both of them (first removing the second one, to test
    # `Selection.__eq__` -- otherwise it is bypassed as `list.remove`
    # checks identity first).
    _process_event(*_get_remove_args(cursor.selections[1]))
    assert len(cursor.selections) == len(ax.figure.artists) == 1
    _process_event(*_get_remove_args(cursor.selections[0]))
    assert len(cursor.selections) == len(ax.figure.artists) == 0
    # Will project on the vertex at (.2, .8).
    _process_event("__mouse_click__", ax, (.2 - .001, .8 + .001), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == 1


@pytest.mark.parametrize("plotter",
                         [lambda ax, *args: ax.plot(*args, ls="", marker="o"),
                          Axes.scatter])
def test_scatter(ax, plotter):
    plotter(ax, [0, .5, 1], [0, .5, 1])
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (.2, .2), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == 0
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == 1


def test_scatter_text(ax):
    ax.scatter([0, 1], [0, 1], c=[2, 3])
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (0, 0), 1)
    assert _parse_annotation(
        cursor.selections[0], "x=(.*)\ny=(.*)\n\[(.*)\]") == (0, 0, 2)


def test_steps_index():
    index = _pick_info.Index(0, .5, .5)
    assert np.floor(index) == 0 and np.ceil(index) == 1
    assert str(index) == "0.(x=0.5, y=0.5)"


def test_steps_pre(ax):
    ax.plot([0, 1], [0, 1], drawstyle="steps-pre")
    ax.set(xlim=(-1, 2), ylim=(-1, 2))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (1, 0), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (0, .5), 1)
    index = cursor.selections[0].index
    assert (index.int, index.x, index.y) == approx((0, 0, .5))
    _process_event("__mouse_click__", ax, (.5, 1), 1)
    index = cursor.selections[0].index
    assert (index.int, index.x, index.y) == approx((0, .5, 1))


def test_steps_mid(ax):
    ax.plot([0, 1], [0, 1], drawstyle="steps-mid")
    ax.set(xlim=(-1, 2), ylim=(-1, 2))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (0, 1), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (1, 0), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (.25, 0), 1)
    index = cursor.selections[0].index
    assert (index.int, index.x, index.y) == approx((0, .25, 0))
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    index = cursor.selections[0].index
    assert (index.int, index.x, index.y) == approx((0, .5, .5))
    _process_event("__mouse_click__", ax, (.75, 1), 1)
    index = cursor.selections[0].index
    assert (index.int, index.x, index.y) == approx((0, .75, 1))


def test_steps_post(ax):
    ax.plot([0, 1], [0, 1], drawstyle="steps-post")
    ax.set(xlim=(-1, 2), ylim=(-1, 2))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (0, 1), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (.5, 0), 1)
    index = cursor.selections[0].index
    assert (index.int, index.x, index.y) == approx((0, .5, 0))
    _process_event("__mouse_click__", ax, (1, .5), 1)
    index = cursor.selections[0].index
    assert (index.int, index.x, index.y) == approx((0, 1, .5))


@pytest.mark.parametrize("ls", ["-", "o"])
def test_line_single_point(ax, ls):
    ax.plot(0, ls)
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (.001, .001), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == (ls == "o")
    if cursor.selections:
        assert tuple(cursor.selections[0].target) == (0, 0)


@pytest.mark.parametrize("plot_args,click,targets",
                         [(([0, 1, np.nan, 3, 4],), (.5, .5), [(.5, .5)]),
                          (([np.nan, np.nan],), (0, 0), []),
                          (([np.nan, np.nan], "."), (0, 0), [])])
def test_nan(ax, plot_args, click, targets):
    ax.plot(*plot_args)
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, click, 1)
    assert len(cursor.selections) == len(ax.figure.artists) == len(targets)
    for sel, target in zip(cursor.selections, targets):
        assert sel.target == approx(target)


def test_repeated_point(ax):
    ax.plot([0, 1, 1, 2], [0, 1, 1, 2])
    cursor = mplcursors.cursor()
    with pytest.warns(None) as record:
        _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert not _internal_warnings(record)


@pytest.mark.parametrize("origin", ["lower", "upper"])
def test_image(ax, origin):
    array = np.arange(6).reshape((3, 2))
    ax.imshow(array, origin=origin)

    cursor = mplcursors.cursor()
    # Annotation text includes image value.
    _process_event("__mouse_click__", ax, (.25, .25), 1)
    sel, = cursor.selections
    assert _parse_annotation(
        sel, r"x=(.*)\ny=(.*)\n\[0\]") == approx((.25, .25))
    # Moving around.
    _process_event("key_press_event", ax, (.123, .456), "shift+right")
    sel, = cursor.selections
    assert _parse_annotation(sel, r"x=(.*)\ny=(.*)\n\[1\]") == (1, 0)
    assert array[sel.index] == 1
    _process_event("key_press_event", ax, (.123, .456), "shift+right")
    sel, = cursor.selections
    assert _parse_annotation(sel, r"x=(.*)\ny=(.*)\n\[0\]") == (0, 0)
    assert array[sel.index] == 0
    _process_event("key_press_event", ax, (.123, .456), "shift+up")
    sel, = cursor.selections
    assert (_parse_annotation(sel, r"x=(.*)\ny=(.*)\n\[(.*)\]")
            == {"upper": (0, 2, 4), "lower": (0, 1, 2)}[origin])
    assert array[sel.index] == {"upper": 4, "lower": 2}[origin]
    _process_event("key_press_event", ax, (.123, .456), "shift+down")
    sel, = cursor.selections
    assert _parse_annotation(sel, r"x=(.*)\ny=(.*)\n\[0\]") == (0, 0)
    assert array[sel.index] == 0

    cursor = mplcursors.cursor()
    # Not picking out-of-axes or of image.
    _process_event("__mouse_click__", ax, (-1, -1), 1)
    assert len(cursor.selections) == 0
    ax.set(xlim=(-1, None), ylim=(-1, None))
    _process_event("__mouse_click__", ax, (-.75, -.75), 1)
    assert len(cursor.selections) == 0


def test_image_rgb(ax):
    ax.imshow([[[.1, .2, .3], [.4, .5, .6]]])
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (0, 0), 1)
    sel, = cursor.selections
    assert _parse_annotation(
        sel, r"x=(.*)\ny=(.*)\n\[0.1, 0.2, 0.3\]") == approx((0, 0))
    _process_event("key_press_event", ax, (.123, .456), "shift+right")
    sel, = cursor.selections
    assert _parse_annotation(
        sel, r"x=(.*)\ny=(.*)\n\[0.4, 0.5, 0.6\]") == approx((1, 0))


def test_image_subclass(ax):
    # Cannot move around `PcolorImage`s.
    ax.pcolorfast(np.arange(3) ** 2, np.arange(3) ** 2, np.zeros((2, 2)))
    cursor = mplcursors.cursor()
    with pytest.warns(UserWarning):
        _process_event("__mouse_click__", ax, (1, 1), 1)
    assert len(cursor.selections) == 0


def test_linecollection(ax):
    ax.eventplot([])  # This must not raise a division by len([]) == 0.
    ax.eventplot([0, 1])
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (0, 0), 1)
    _process_event("__mouse_click__", ax, (.5, 1), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (0, 1), 1)
    assert cursor.selections[0].index == approx((0, .5))


def test_patchcollection(ax):
    ax.add_collection(mpl.collections.PatchCollection([
        mpl.patches.Rectangle(xy, .1, .1) for xy in [(0, 0), (.5, .5)]]))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (.05, .05), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (.6, .6), 1)
    # The precision is really bad :(
    assert cursor.selections[0].index == approx((1, 2), abs=2e-2)


@pytest.mark.parametrize("plotter", [Axes.quiver, Axes.barbs])
def test_quiver_and_barbs(ax, plotter):
    plotter(ax, range(3), range(3))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (.5, 0), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (1, 0), 1)
    assert _parse_annotation(
        cursor.selections[0], r"x=(.*)\ny=(.*)\n\(1, 1\)") == (1, 0)


@pytest.mark.parametrize("plotter,order",
                         [(Axes.bar, np.s_[:]), (Axes.barh, np.s_[::-1])])
def test_bar(ax, plotter, order):
    container = plotter(ax, range(3), range(1, 4))
    cursor = mplcursors.cursor()
    assert len(cursor.artists) == 1
    _process_event("__mouse_click__", ax, (0, 2)[order], 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (0, .5)[order], 1)
    assert cursor.selections[0].artist is container  # not the ContainerArtist.
    assert cursor.selections[0].target == approx((0, 1)[order])


def test_errorbar(ax):
    ax.errorbar(range(2), range(2), [(1, 1), (1, 2)])
    cursor = mplcursors.cursor()
    assert len(cursor.artists) == 1
    _process_event("__mouse_click__", ax, (0, 2), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert cursor.selections[0].target == approx((.5, .5))
    assert _parse_annotation(
        cursor.selections[0], "x=(.*)\ny=(.*)") == approx((.5, .5))
    _process_event("__mouse_click__", ax, (0, 1), 1)
    assert cursor.selections[0].target == approx((0, 0))
    assert _parse_annotation(
        cursor.selections[0], r"x=(.*)\ny=\$(.*)\\pm(.*)\$") == (0, 0, 1)
    _process_event("__mouse_click__", ax, (1, 2), 1)
    sel, = cursor.selections
    assert sel.target == approx((1, 1))
    assert _parse_annotation(
        sel, r"x=(.*)\ny=\$(.*)_\{(.*)\}\^\{(.*)\}\$") == (1, 1, -1, 2)


def test_dataless_errorbar(ax):
    # Unfortunately, the original data cannot be recovered when fmt="none".
    ax.errorbar(range(2), range(2), [(1, 1), (1, 2)], fmt="none")
    cursor = mplcursors.cursor()
    assert len(cursor.artists) == 1
    _process_event("__mouse_click__", ax, (0, 0), 1)
    assert len(cursor.selections) == 0


def test_stem(ax):
    with pytest.warns(None):  # stem use_line_collection API change.
        ax.stem([1, 2, 3], use_line_collection=True)
    cursor = mplcursors.cursor()
    assert len(cursor.artists) == 1
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == 0
    _process_event("__mouse_click__", ax, (0, 1), 1)
    assert cursor.selections[0].target == approx((0, 1))
    _process_event("__mouse_click__", ax, (0, .5), 1)
    assert cursor.selections[0].target == approx((0, 1))


@pytest.mark.parametrize(
    "plotter,warns",
    [(lambda ax: ax.text(.5, .5, "foo"), False),
     (lambda ax: ax.fill_between([0, 1], [0, 1]), True)])
def test_misc_artists(ax, plotter, warns):
    plotter(ax)
    cursor = mplcursors.cursor()
    with pytest.warns(None) as record:
        _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == 0
    assert len(_internal_warnings(record)) == warns


def test_indexless_projections(fig):
    ax = fig.subplots(subplot_kw={"projection": "polar"})
    ax.plot([1, 2], [3, 4])
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (1, 3), 1)
    assert len(cursor.selections) == 1
    _process_event("key_press_event", ax, (.123, .456), "shift+left")


def test_cropped_by_axes(fig):
    axs = fig.subplots(2)
    axs[0].plot([0, 0], [0, 1])
    # Pan to hide the line behind the second axes.
    axs[0].set(xlim=(-1, 1), ylim=(1, 2))
    axs[1].set(xlim=(-1, 1), ylim=(-1, 1))
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", axs[1], (0, 0), 1)
    assert len(cursor.selections) == 0


@pytest.mark.parametrize("plotter", [Axes.plot, Axes.scatter, Axes.errorbar])
def test_move(ax, plotter):
    plotter(ax, [0, 1, 2], [0, 1, np.nan])
    cursor = mplcursors.cursor()
    # Nothing happens with no cursor.
    _process_event("key_press_event", ax, (.123, .456), "shift+left")
    assert len(cursor.selections) == 0
    # Now we move the cursor left or right.
    if plotter in [Axes.plot, Axes.errorbar]:
        _process_event("__mouse_click__", ax, (.5, .5), 1)
        assert tuple(cursor.selections[0].target) == approx((.5, .5))
        _process_event("key_press_event", ax, (.123, .456), "shift+up")
        _process_event("key_press_event", ax, (.123, .456), "shift+left")
    elif plotter is Axes.scatter:
        _process_event("__mouse_click__", ax, (0, 0), 1)
        _process_event("key_press_event", ax, (.123, .456), "shift+up")
    assert tuple(cursor.selections[0].target) == (0, 0)
    assert cursor.selections[0].index == 0
    _process_event("key_press_event", ax, (.123, .456), "shift+right")
    assert tuple(cursor.selections[0].target) == (1, 1)
    assert cursor.selections[0].index == 1
    # Skip through nan.
    _process_event("key_press_event", ax, (.123, .456), "shift+right")
    assert tuple(cursor.selections[0].target) == (0, 0)
    assert cursor.selections[0].index == 0


@pytest.mark.parametrize(
    "hover", [True, HoverMode.Persistent, 2, HoverMode.Transient])
def test_hover(ax, hover):
    l1, = ax.plot([0, 1])
    l2, = ax.plot([1, 2])
    cursor = mplcursors.cursor(hover=hover)
    _process_event("motion_notify_event", ax, (.5, .5), 1)
    assert len(cursor.selections) == 0  # No trigger if mouse button pressed.
    _process_event("motion_notify_event", ax, (.5, .5))
    assert cursor.selections[0].artist == l1
    _process_event("motion_notify_event", ax, (.5, 1))
    assert bool(cursor.selections) == (hover == HoverMode.Persistent)
    _process_event("motion_notify_event", ax, (.5, 1.5))
    assert cursor.selections[0].artist == l2


@pytest.mark.parametrize("plotter", [Axes.plot, Axes.scatter])
def test_highlight(ax, plotter):
    plotter(ax, [0, 1], [0, 1])
    ax.set(xlim=(-1, 2), ylim=(-1, 2))
    base_children = {*ax.artists, *ax.lines, *ax.collections}
    cursor = mplcursors.cursor(highlight=True)
    _process_event("__mouse_click__", ax, (0, 0), 1)
    # On Matplotlib<=3.4, the highlight went to ax.artists.  On >=3.5, it goes
    # to its type-specific container.  The construct below handles both cases.
    assert [*{*ax.artists, *ax.lines, *ax.collections} - base_children] \
        == cursor.selections[0].extras != []
    _process_event(*_get_remove_args(cursor.selections[0]))
    assert len({*ax.artists, *ax.lines, *ax.collections} - base_children) == 0


def test_misc_artists_highlight(ax):
    # Unsupported artists trigger a warning upon a highlighting attempt.
    ax.imshow([[0, 1], [2, 3]])
    cursor = mplcursors.cursor(highlight=True)
    with pytest.warns(UserWarning):
        _process_event("__mouse_click__", ax, (.5, .5), 1)


def test_callback(ax):
    ax.plot([0, 1])
    calls = []
    cursor = mplcursors.cursor()
    @cursor.connect("add")
    def on_add(sel):
        calls.append(sel)
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(calls) == 1
    cursor.disconnect("add", on_add)
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(calls) == 1


def test_remove_while_adding(ax):
    ax.plot([0, 1])
    cursor = mplcursors.cursor()
    cursor.connect("add", cursor.remove_selection)
    _process_event("__mouse_click__", ax, (.5, .5), 1)


def test_no_duplicate(ax):
    ax.plot([0, 1])
    cursor = mplcursors.cursor(multiple=True)
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == 1


def test_remove_multiple_overlapping(ax):
    ax.plot([0, 1])
    cursor = mplcursors.cursor(multiple=True)
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    sel, = cursor.selections
    cursor.add_selection(copy.copy(sel))
    assert len(cursor.selections) == 2
    _process_event(*_get_remove_args(sel))
    assert [*map(id, cursor.selections)] == [id(sel)]  # To check LIFOness.
    _process_event(*_get_remove_args(sel))
    assert len(cursor.selections) == 0


def test_autoalign(ax):
    ax.plot([0, 1])
    cursor = mplcursors.cursor()
    cursor.connect(
        "add", lambda sel: sel.annotation.set(position=(-10, 0)))
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    sel, = cursor.selections
    assert (sel.annotation.get_ha() == "right"
            and sel.annotation.get_va() == "center")
    cursor.remove_selection(sel)
    cursor.connect(
        "add", lambda sel: sel.annotation.set(ha="center", va="bottom"))
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    sel, = cursor.selections
    assert (sel.annotation.get_ha() == "center"
            and sel.annotation.get_va() == "bottom")


@pytest.mark.xfail(
    int(mpl.__version__.split(".")[0]) < 3,
    reason="Matplotlib fails to disconnect dragging callbacks.")
def test_drag(ax, capsys):
    l, = ax.plot([0, 1])
    cursor = mplcursors.cursor()
    cursor.connect(
        "add", lambda sel: sel.annotation.set(position=(0, 0)))
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    _process_event("button_press_event", ax, (.5, .5), 1)
    _process_event("motion_notify_event", ax, (.4, .6), 1)
    assert not capsys.readouterr().err


def test_removed_artist(ax):
    l, = ax.plot([0, 1])
    cursor = mplcursors.cursor()
    l.remove()
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == 0


def test_remove_cursor(ax):
    ax.plot([0, 1])
    cursor = mplcursors.cursor()
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == 1
    cursor.remove()
    assert len(cursor.selections) == len(ax.figure.artists) == 0
    _process_event("__mouse_click__", ax, (.5, .5), 1)
    assert len(cursor.selections) == len(ax.figure.artists) == 0


def test_keys(ax):
    ax.plot([0, 1])
    cursor = mplcursors.cursor(multiple=True)
    _process_event("__mouse_click__", ax, (.3, .3), 1)
    # Toggle visibility.
    _process_event("key_press_event", ax, (.123, .456), "v")
    assert not cursor.selections[0].annotation.get_visible()
    _process_event("key_press_event", ax, (.123, .456), "v")
    assert cursor.selections[0].annotation.get_visible()
    # Disable the cursor.
    _process_event("key_press_event", ax, (.123, .456), "e")
    assert not cursor.enabled
    # (Adding becomes inactive.)
    _process_event("__mouse_click__", ax, (.6, .6), 1)
    assert len(cursor.selections) == 1
    # (Removing becomes inactive.)
    ax.figure.canvas.draw()
    _process_event(*_get_remove_args(cursor.selections[0]))
    assert len(cursor.selections) == 1
    # (Moving becomes inactive.)
    old_target = cursor.selections[0].target
    _process_event("key_press_event", ax, (.123, .456), "shift+left")
    new_target = cursor.selections[0].target
    assert (old_target == new_target).all()
    # Reenable it.
    _process_event("key_press_event", ax, (.123, .456), "e")
    assert cursor.enabled
    _process_event(*_get_remove_args(cursor.selections[0]))
    assert len(cursor.selections) == 0


def test_convenience(ax):
    l, = ax.plot([1, 2])
    assert len(mplcursors.cursor().artists) == 1
    assert len(mplcursors.cursor(ax).artists) == 1
    assert len(mplcursors.cursor(l).artists) == 1
    assert len(mplcursors.cursor([l]).artists) == 1
    bc = ax.bar(range(3), range(3))
    assert len(mplcursors.cursor(bc).artists) == 1


def test_invalid_args():
    pytest.raises(ValueError, mplcursors.cursor,
                  bindings={"foo": 42})
    pytest.raises(ValueError, mplcursors.cursor,
                  bindings={"select": 1, "deselect": 1})
    pytest.raises(ValueError, mplcursors.cursor().connect,
                  "foo")


def test_multiple_figures(ax):
    ax1 = ax
    _, ax2 = plt.subplots()
    ax1.plot([0, 1])
    ax2.plot([0, 1])
    cursor = mplcursors.cursor([ax1, ax2], multiple=True)
    # Add something on the first axes.
    _process_event("__mouse_click__", ax1, (.5, .5), 1)
    assert len(cursor.selections) == 1
    assert len(ax1.figure.artists) == 1
    assert len(ax2.figure.artists) == 0
    # Right-clicking on the second axis doesn't remove it.
    remove_args = [*_get_remove_args(cursor.selections[0])]
    remove_args[remove_args.index(ax1)] = ax2
    _process_event(*remove_args)
    assert len(cursor.selections) == 1
    assert len(ax1.figure.artists) == 1
    assert len(ax2.figure.artists) == 0
    # Remove it, add something on the second.
    _process_event(*_get_remove_args(cursor.selections[0]))
    _process_event("__mouse_click__", ax2, (.5, .5), 1)
    assert len(cursor.selections) == 1
    assert len(ax1.figure.artists) == 0
    assert len(ax2.figure.artists) == 1


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


@pytest.mark.parametrize(
    "example",
    [path for path in Path("examples").glob("*.py")
     if "test: skip" not in path.read_text()])
def test_example(example):
    subprocess.check_call(
        [sys.executable, "-mexamples.{}".format(example.with_suffix("").name)],
        # Unset $DISPLAY to avoid the non-GUI backend warning.
        env={**os.environ, "DISPLAY": "", "MPLBACKEND": "Agg"})
