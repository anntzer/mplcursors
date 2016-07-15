from collections import namedtuple
import copy
from weakref import WeakKeyDictionary

from . import _contains_test


__all__ = ["Cursor"]


def _default_fmt(d):
    ax = d["ax"]
    x = d["x"]
    y = d["y"]
    return "x: {}\ny: {}".format(ax.format_xdata(x), ax.format_ydata(y))


_default_annotation_kwargs = dict(
    xytext=(-15, 15), textcoords="offset points",
    bbox=dict(boxstyle="round,pad=.5", fc="yellow", alpha=.5, ec="k"),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", ec="k")
)
_default_highlight_kwargs = dict(c="yellow", mec="yellow", lw=3, mew=3)


def _reassigned_axes_event(event, ax):
    """Reassign `event` to `ax`.
    """
    event = copy.copy(event)
    event.xdata, event.ydata = (
        ax.transData.inverted().transform_point((event.x, event.y)))
    return event


_Selection = namedtuple("_Selection", "annotation highlight")


class Cursor:
    _keep_alive = WeakKeyDictionary()

    def __init__(self,
                 artists,
                 *,
                 multiple=False,
                 format=_default_fmt,
                 annotation_kwargs=None,
                 highlight=False,
                 display_button=1,
                 hide_button=3):

        self._artists = artists
        self._multiple = multiple
        self._format = format
        self._annotation_kwargs = (
            _default_annotation_kwargs if annotation_kwargs is None
            else annotation_kwargs)
        self._highlight_kwargs = (
            None if highlight is False
            else _default_highlight_kwargs if highlight is True
            else highlight
        )
        if display_button == hide_button:
            raise ValueError(
                "`display_button` and `hide_button` must be different")
        self._display_button = display_button
        self._hide_button = hide_button

        self._figures = {artist.figure for artist in artists}
        self._axes = {artist.axes for artist in artists}

        for figure in self._figures:
            type(self)._keep_alive.setdefault(figure, []).append(self)
            figure.canvas.mpl_connect(
                "button_press_event", self._on_button_press)

        self._selections = []

    def _on_button_press(self, event):
        if event.button == self._display_button:
            self._on_display_button_press(event)
        if event.button == self._hide_button:
            self._on_hide_button_press(event)

    def _on_display_button_press(self, event):
        # Work around lack of support for twinned axes.
        per_axes_event = {ax: _reassigned_axes_event(event, ax)
                          for ax in self._axes}
        containments = []
        for artist in self._artists:
            if event.canvas is not artist.figure.canvas:
                continue
            containment = _contains_test.contains(
                artist, per_axes_event[artist.axes])
            if containment:
                containments.append((containment, artist))
        if not containments:
            return
        containment, artist = min(containments)
        ax = artist.axes
        ann = ax.annotate(
            self._format(containment.info),
            xy=containment.target,
            **self._annotation_kwargs)
        if self._highlight_kwargs is not None:
            hl = copy.copy(artist)
            hl.set(**self._highlight_kwargs)
            ax.add_artist(hl)
        if not self._multiple:
            while self._selections:
                self._remove_selection(self._selections[-1])
        self._selections.append(_Selection(ann, hl))
        ax.figure.canvas.draw_idle()

    def _on_hide_button_press(self, event):
        for sel in self._selections:
            ann = sel.annotation
            if event.canvas is not ann.figure.canvas:
                continue
            contained, _ = ann.contains(event)
            if contained:
                self._remove_selection(sel)

    def _remove_selection(self, sel):
        self._selections.remove(sel)
        ann = sel.annotation
        hl = sel.highlight
        ax = ann.axes
        ax.texts.remove(ann)
        if hl is not None:
            ax.artists.remove(hl)
        ax.figure.canvas.draw_idle()
