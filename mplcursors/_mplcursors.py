from collections import namedtuple
import copy
from types import MappingProxyType
from weakref import WeakKeyDictionary

from matplotlib.cbook import CallbackRegistry

from . import _pick_info


__all__ = ["Cursor", "default_annotation_kwargs", "default_highlight_kwargs"]


default_annotation_kwargs = MappingProxyType(dict(
    xytext=(-15, 15), textcoords="offset points",
    bbox=dict(boxstyle="round,pad=.5", fc="yellow", alpha=.5, ec="k"),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", shrinkB=0, ec="k")))
default_highlight_kwargs = MappingProxyType(dict(
    c="yellow", mec="yellow", lw=3, mew=3))


def _reassigned_axes_event(event, ax):
    """Reassign `event` to `ax`.
    """
    event = copy.copy(event)
    event.xdata, event.ydata = (
        ax.transData.inverted().transform_point((event.x, event.y)))
    return event


_Selection = namedtuple("_Selection", "pick_info annotation extras")


class Cursor:
    _keep_alive = WeakKeyDictionary()

    def __init__(self,
                 artists,
                 *,
                 hover=False,
                 multiple=False,
                 transformer=lambda c: c,
                 annotation_kwargs=None,
                 draggable=False,
                 highlight=False,
                 display_button=1,
                 hide_button=3):

        self._artists = artists
        self._multiple = multiple
        self._transformer = transformer
        self._annotation_kwargs = dict(
            default_annotation_kwargs, **annotation_kwargs or {})
        self._draggable = draggable
        self._highlight_kwargs = (
            None if highlight is False
            else default_highlight_kwargs if highlight is True
            else dict(default_annotation_kwargs, **highlight)
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
            if hover:
                if multiple:
                    raise ValueError("`hover` and `multiple` are incompatible")
                figure.canvas.mpl_connect(
                    "motion_notify_event", self._on_display_button_press)
            else:
                figure.canvas.mpl_connect(
                    "button_press_event", self._on_button_press)

        self._selections = []
        self._callbacks = CallbackRegistry()

    @property
    def callbacks(self):
        return self._callbacks

    def add_annotation(self, pick_info):
        ann = pick_info.artist.axes.annotate(
            pick_info.ann_text, xy=pick_info.target, **self._annotation_kwargs)
        if self._draggable:
            ann.draggable()
        return ann

    def add_highlight(self, artist):
        hl = copy.copy(artist)
        hl.set(**self._highlight_kwargs)
        artist.axes.add_artist(hl)
        return hl

    def _on_button_press(self, event):
        if event.button == self._display_button:
            self._on_display_button_press(event)
        if event.button == self._hide_button:
            self._on_hide_button_press(event)

    def _on_display_button_press(self, event):
        # Work around lack of support for twinned axes.
        per_axes_event = {ax: _reassigned_axes_event(event, ax)
                          for ax in self._axes}
        pis = []
        for artist in self._artists:
            if event.canvas is not artist.figure.canvas:
                continue
            pi = _pick_info.compute_pick(artist, per_axes_event[artist.axes])
            if pi:
                pis.append(pi)
        if not pis:
            return
        pi = self._transformer(min(pis, key=lambda c: c.dist))
        artist = pi.artist
        ax = artist.axes
        ann = self.add_annotation(pi)
        extras = []
        if self._highlight_kwargs is not None:
            extras.append(self.add_highlight(artist))
        if not self._multiple:
            while self._selections:
                self._remove_selection(self._selections[-1])
        sel = _Selection(pi, ann, extras)
        self._selections.append(sel)
        self._callbacks.process("add", sel)
        ax.figure.canvas.draw_idle()

    def _on_hide_button_press(self, event):
        for sel in self._selections:
            ann = sel.annotation
            if event.canvas is not ann.figure.canvas:
                continue
            contained, _ = ann.contains(event)
            if contained:
                self._remove_selection(sel)
                self._callbacks.process("remove", sel)
            ax.axes.figure.canvas.draw_idle()

    def _remove_selection(self, sel):
        self._selections.remove(sel)
        sel.annotation.remove()
        for artist in sel.extras:
            artist.remove()
