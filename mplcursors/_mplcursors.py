from collections import namedtuple
from collections.abc import Iterable
import copy
from functools import partial
from types import MappingProxyType
import warnings
from weakref import WeakKeyDictionary

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cbook import CallbackRegistry

from . import _pick_info
from ._pick_info import PickInfo


default_annotation_kwargs = MappingProxyType(dict(
    xytext=(-15, 15),
    textcoords="offset points",
    bbox=dict(
        boxstyle="round,pad=.5",
        fc="yellow",
        alpha=.5,
        ec="k"),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3",
        shrinkB=0,
        ec="k")))
default_highlight_kwargs = MappingProxyType(dict(
    c="yellow",
    mec="yellow",
    lw=3,
    mew=3))
default_bindings = MappingProxyType(dict(
    select=1,
    deselect=3,
    previous="shift+left",
    next="shift+right",
    toggle_visibility="d",
    toggle_enabled="t"))


def _reassigned_axes_event(event, ax):
    """Reassign `event` to `ax`.
    """
    event = copy.copy(event)
    event.xdata, event.ydata = (
        ax.transData.inverted().transform_point((event.x, event.y)))
    return event


Selection = namedtuple(
    "Selection", PickInfo._fields + ("annotation", "extras"))
for _field in PickInfo._fields:
    getattr(Selection, _field).__doc__ = getattr(PickInfo, _field).__doc__
del _field
Selection.annotation.__doc__ = "The instantiated `matplotlib.text.Annotation`."
Selection.extras.__doc__ = (
    "An additional list of artists (e.g., highlighters) that will be cleared "
    "at the same time as the annotation.")


def _selection_to_pick_info(sel):
    return PickInfo(sel[:len(PickInfo._fields)])


class Cursor:
    """A cursor for selecting artists on a matplotlib figure.
    """

    _keep_alive = WeakKeyDictionary()

    def __init__(self,
                 artists,
                 *,
                 multiple=False,
                 highlight=False,
                 hover=False,
                 bindings=default_bindings):
        """Construct a cursor.

        Parameters
        ----------

        artists : List[Artist]
            A list of artists that can be selected by this cursor.

        multiple : bool
            Whether multiple artists can be "on" at the same time (defaults to
            False).

        highlight : bool
            Whether to also highlight the selected artist.  If so,
            "highlighter" artists will be placed as the first item in the
            :attr:`extras` attribute of the `Selection`.

        bindings : dict
            A mapping of button and keybindings to actions.  Valid entries are:

            =================== ===============================================
            'select'            mouse button to select an artist (default: 1)
            'deselect'          mouse button to deselect an artist (default: 3)
            'previous'          move to the previous point in the selected path
                                (default: shift+left)
            'next'              move to the next point in the selected path
                                (default: shift+right)
            'toggle_visibility' toggle visibility of all cursors (default: d)
            'toggle_enabled'    toggle whether the cursor is active
                                (default: t)
            =================== ===============================================

        hover : bool
            Whether to select artists upon hovering instead of by clicking.
        """

        # Copy `artists` to maintain it constant.
        self._artists = artists = list(artists)

        for artist in artists:
            type(self)._keep_alive.setdefault(artist, []).append(self)

        self._multiple = multiple
        self._highlight = highlight

        self._axes = {artist.axes for artist in artists}
        self._enabled = True
        self._selections = []
        self._callbacks = CallbackRegistry()

        connect_pairs = [("key_press_event", self._on_key_press)]
        if hover:
            if multiple:
                raise ValueError("`hover` and `multiple` are incompatible")
            connect_pairs += [
                ("motion_notify_event", self._on_select_button_press)]
        else:
            connect_pairs += [
                ("button_press_event", self._on_button_press)]
        self._disconnect_cids = [
            partial(canvas.mpl_disconnect, canvas.mpl_connect(*pair))
            for pair in connect_pairs
            for canvas in {artist.figure.canvas for artist in artists}]

        bindings = {**default_bindings, **bindings}
        if set(bindings) != set(default_bindings):
            raise ValueError("Unknown bindings")
        actually_bound = {k: v for k, v in bindings.items() if v is not None}
        if len(set(actually_bound.values())) != len(actually_bound):
            raise ValueError("Duplicate bindings")
        self._bindings = bindings

    @property
    def enabled(self):
        """Whether clicks are registered for picking and unpicking events.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    @property
    def artists(self):
        """The tuple of selectable artists.
        """
        return tuple(self._artists)

    @property
    def selections(self):
        """The tuple of current `Selection`\\s.
        """
        return tuple(self._selections)

    def add_annotation(self, pick_info):
        """Add an annotation from a `PickInfo`.

        Returns a `Selection` wrapping the `PickInfo` and the added artists;
        emits the ``"add"`` event with the `Selection` as argument.
        """
        ann = pick_info.artist.axes.annotate(
            _pick_info.get_ann_text(*pick_info),
            xy=pick_info.target,
            **default_annotation_kwargs)
        ann.draggable(use_blit=True)
        extras = []
        if self._highlight:
            extras.append(self.add_highlight(pick_info.artist))
        if not self._multiple:
            while self._selections:
                self._remove_selection(self._selections[-1])
        sel = Selection(*pick_info, ann, extras)
        self._selections.append(sel)
        self._callbacks.process("add", sel)
        pick_info.artist.figure.canvas.draw_idle()
        return ann

    def add_highlight(self, artist):
        """Create, add and return a highlighting artist.

        It is up to the caller to register the artist with the proper
        `Selection` in order to ensure cleanup upon deselection.
        """
        hl = copy.copy(artist)
        hl.set(**default_highlight_kwargs)
        artist.axes.add_artist(hl)
        return hl

    def connect(self, event, func=None):
        """Connect a callback to a `Cursor` event; return the callback id.

        Two classes of event can be emitted, both with a `Selection` as single
        argument:

            - ``"add"`` when a `Selection` is added, and
            - ``"remove"`` when a `Selection` is removed.

        The callback registry relies on :mod:`matplotlib`'s implementation; in
        particular, only weak references are kept for bound methods.

        This method is can also be used as a decorator::

            @cursor.connect("add")
            def on_add(sel):
                ...
        """
        if event not in ["add", "remove"]:
            raise ValueError("Invalid cursor event: {}".format(event))
        if func is None:
            return partial(self.connect, event)
        return self._callbacks.connect(event, func)

    def disconnect(self, cid):
        """Disconnect a previously connected callback id.
        """
        self._callbacks.disconnect(cid)

    def clear(self):
        """Remove all `Selection`\\s and disconnect all callbacks.
        """
        for disconnect_cid in self._disconnect_cids:
            disconnect_cid()
        while self._selections:
            self._remove_selection(self._selections[-1])

    def _on_button_press(self, event):
        if event.button == self._bindings["select"]:
            self._on_select_button_press(event)
        if event.button == self._bindings["deselect"]:
            self._on_deselect_button_press(event)

    def _on_select_button_press(self, event):
        if event.canvas.widgetlock.locked() or not self.enabled:
            return
        # Work around lack of support for twinned axes.
        per_axes_event = {ax: _reassigned_axes_event(event, ax)
                          for ax in self._axes}
        pis = []
        for artist in self._artists:
            # e.g., removed artist.
            if artist.axes is None:
                continue
            if event.canvas is not artist.figure.canvas:
                continue
            try:
                pi = _pick_info.compute_pick(
                    artist, per_axes_event[artist.axes])
            except NotImplementedError as e:
                warnings.warn(str(e))
            if pi:
                pis.append(pi)
        if not pis:
            return
        self.add_annotation(min(pis, key=lambda pi: pi.dist))

    def _on_deselect_button_press(self, event):
        if event.canvas.widgetlock.locked() or not self.enabled:
            return
        for sel in self._selections:
            ann = sel.annotation
            if event.canvas is not ann.figure.canvas:
                continue
            contained, _ = ann.contains(event)
            if contained:
                self._remove_selection(sel)

    def _on_key_press(self, event):
        if event.key == self._bindings["toggle_enabled"]:
            self.enabled = not self.enabled
        elif event.key == self._bindings["toggle_visibility"]:
            for sel in self._selections:
                sel.annotation.set_visible(not sel.annotation.get_visible())
                sel.annotation.figure.canvas.draw_idle()
        if self._selections:
            sel = self._selections[-1]
        else:
            return
        if event.key == self._bindings["previous"]:
            self._remove_selection(sel)
            self.add_annotation(
                _pick_info.move(*_selection_to_pick_info(sel), -1))
        elif event.key == self._bindings["next"]:
            self._remove_selection(sel)
            self.add_annotation(
                _pick_info.move(*_selection_to_pick_info(sel), 1))

    def _remove_selection(self, sel):
        self._selections.remove(sel)
        sel.annotation.figure.canvas.draw_idle()
        # Work around matplotlib/matplotlib#6785.
        draggable = sel.annotation._draggable
        if draggable:
            draggable.disconnect()
            try:
                c = draggable._c1
            except AttributeError:
                pass
            else:
                draggable.canvas.mpl_disconnect(draggable._c1)
        # (end of workaround).
        sel.annotation.remove()
        for artist in sel.extras:
            artist.figure.canvas.draw_idle()
            artist.remove()
        self._callbacks.process("remove", sel)


def cursor(artists_or_axes=None, **kwargs):
    """Create a :class:`Cursor` for a list of artists or axes.

    Parameters
    ----------

    artists_or_axes : Optional[List[Union[Artist, Axes]]]
        All artists in the list and all artists on any of the axes passed in
        the list are selectable by the constructed :class:`Cursor`.  Defaults
        to all artists on any of the figures that :mod:`pyplot` is tracking.

    **kwargs
        Keyword arguments are passed to the :class:`Cursor` constructor.
    """

    if artists_or_axes is None:
        artists_or_axes = [ax
                           for fig in map(plt.figure, plt.get_fignums())
                           for ax in fig.axes]
    elif not isinstance(artists_or_axes, Iterable):
        artists_or_axes = [artists_or_axes]
    artists = []
    for entry in artists_or_axes:
        if isinstance(entry, Axes):
            ax = entry
            artists.extend(ax.lines + ax.patches + ax.collections + ax.images)
            for container in ax.containers:
                artists.extend(container)
        else:
            artist = entry
            artists.append(artist)
    return Cursor(artists, **kwargs)
