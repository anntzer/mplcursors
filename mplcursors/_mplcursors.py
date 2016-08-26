from collections.abc import Iterable
from contextlib import suppress
import copy
from functools import partial
from types import MappingProxyType
import weakref
from weakref import WeakKeyDictionary

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cbook import CallbackRegistry

from . import _pick_info


default_annotation_kwargs = MappingProxyType(dict(
    xytext=(-15, 15), ha="right", va="bottom",
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
    # Only the kwargs corresponding to properties of the artist will be passed.
    # Line2D.
    color="yellow",
    markeredgecolor="yellow",
    linewidth=3,
    markeredgewidth=3,
    # PathCollection.
    facecolor="yellow",
    edgecolor="yellow"))
default_bindings = MappingProxyType(dict(
    select=1,
    deselect=3,
    left="shift+left",
    right="shift+right",
    up="shift+up",
    down="shift+down",
    toggle_visibility="d",
    toggle_enabled="t"))


def _reassigned_axes_event(event, ax):
    """Reassign `event` to `ax`.
    """
    event = copy.copy(event)
    event.xdata, event.ydata = (
        ax.transData.inverted().transform_point((event.x, event.y)))
    return event


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
            'left'              move to the previous point in the selected
                                path, or to the left in the selected image
                                (default: shift+left)
            'right'             move to the next point in the selected path, or
                                to the right in the selected image
                                (default: shift+right)
            'up'                move up in the selected image
                                (default: shift+up)
            'down'              move down in the selected image
                                (default: shift+down)
            'toggle_visibility' toggle visibility of all cursors (default: d)
            'toggle_enabled'    toggle whether the cursor is active
                                (default: t)
            =================== ===============================================

        hover : bool
            Whether to select artists upon hovering instead of by clicking.
        """

        artists = list(artists)
        # Be careful with GC.
        self._artists = [weakref.ref(artist) for artist in artists]

        for artist in artists:
            type(self)._keep_alive.setdefault(artist, []).append(self)

        self._multiple = multiple
        self._highlight = highlight

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
        # Unfortunately, see matplotlib/matplotlib#6982: `cla()` does not clear
        # `.axes`.
        return tuple(artist for artist in (ref() for ref in self._artists)
                     if artist and artist.axes)

    @property
    def selections(self):
        """The tuple of current `Selection`\\s.
        """
        return tuple(self._selections)

    def add_selection(self, pi):
        """Create an annotation for a `Selection` and register it.

        Returns a new `Selection`, that has been registered by the `Cursor`,
        with the added annotation set in the :attr:`annotation` field and, if
        applicable, the highlighting artist in the :attr:`extras` field.

        Emits the ``"add"`` event with the new `Selection` as argument.
        """
        # pi: "pick_info", i.e. an incomplete selection.
        ann = pi.artist.axes.annotate(
            _pick_info.get_ann_text(*pi),
            xy=pi.target,
            **default_annotation_kwargs)
        ann.draggable(use_blit=True)
        extras = []
        if self._highlight:
            hl = self.add_highlight(*pi)
            if hl:
                extras.append(hl)
        sel = pi._replace(annotation=ann, extras=extras)
        self._selections.append(sel)
        self._callbacks.process("add", sel)
        if (extras
                or len(self._selections) > 1 and not self._multiple
                or not ann.figure.canvas.supports_blit):
            # Either:
            #  - there may be more things to draw, or
            #  - annotation removal will make a full redraw necessary, or
            #  - blitting is not (yet) supported.
            ann.figure.canvas.draw_idle()
        else:
            # Fast path.
            try:
                ann.figure.draw_artist(ann)
                ann.figure.canvas.blit()
            except AttributeError:  # No cached renderer yet.
                ann.figure.canvas.draw_idle()
        # Removal comes after addition so that the fast blitting path works.
        # (This probably also allows weird tricks such as swapping the added
        # selection from a callback.)
        if not self._multiple:
            for sel in self._selections[:-1]:
                self._remove_selection(sel)
        return sel

    def add_highlight(self, artist, *args, **kwargs):
        """Create, add and return a highlighting artist.

        This method is should be called with an "unpacked" `Selection`,
        possibly with some fields set to None.

        It is up to the caller to register the artist with the proper
        `Selection` in order to ensure cleanup upon deselection.
        """
        hl = _pick_info.make_highlight(
            artist, *args,
            **{"highlight_kwargs": default_highlight_kwargs, **kwargs})
        if hl:
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

    def remove(self):
        """Remove all `Selection`\\s and disconnect all callbacks.
        """
        for disconnect_cid in self._disconnect_cids:
            disconnect_cid()
        for sel in self._selections[:]:
            self._remove_selection(sel)

    def _on_button_press(self, event):
        if event.button == self._bindings["select"]:
            self._on_select_button_press(event)
        if event.button == self._bindings["deselect"]:
            self._on_deselect_button_press(event)

    def _filter_mouse_event(self, event):
        # Accept the event iff we are enabled, and either
        #   - no other widget is active, and this is not the second click of a
        #     double click (to prevent double selection), or
        #   - another widget is active, and this is a double click (to bypass
        #     the widget lock).
        return (self.enabled
                and event.canvas.widgetlock.locked() == event.dblclick)

    def _on_select_button_press(self, event):
        if not self._filter_mouse_event(event):
            return
        # Work around lack of support for twinned axes.
        per_axes_event = {ax: _reassigned_axes_event(event, ax)
                          for ax in {artist.axes for artist in self.artists}}
        pis = []
        for artist in self.artists:
            if (artist.axes is None  # Removed or figure-level artist.
                    or event.canvas is not artist.figure.canvas
                    or not artist.axes.contains(event)[0]):  # Cropped by axes.
                continue
            pi = _pick_info.compute_pick(artist, per_axes_event[artist.axes])
            if pi:
                pis.append(pi)
        if not pis:
            return
        self.add_selection(min(pis, key=lambda pi: pi.dist))

    def _on_deselect_button_press(self, event):
        if not self._filter_mouse_event(event):
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
        for key in ["left", "right", "up", "down"]:
            if event.key == self._bindings[key]:
                self._remove_selection(sel)
                self.add_selection(_pick_info.move(*sel, key=key))
                break

    def _remove_selection(self, sel):
        self._selections.remove(sel)
        # Work around matplotlib/matplotlib#6785.
        draggable = sel.annotation._draggable
        try:
            draggable.disconnect()
            sel.annotation.figure.canvas.mpl_disconnect(
                sel.annotation._draggable._c1)
        except AttributeError:
            pass
        # (end of workaround).
        # <artist>.figure will be unset so we save them first.
        figures = {artist.figure for artist in [sel.annotation, *sel.extras]}
        # ValueError is raised if the artist has already been removed.
        with suppress(ValueError):
            sel.annotation.remove()
        for artist in sel.extras:
            with suppress(ValueError):
                artist.remove()
        self._callbacks.process("remove", sel)
        for figure in figures:
            figure.canvas.draw_idle()


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
            artists.extend(
                ax.collections + ax.images + ax.lines + ax.patches + ax.texts)
            # No need to extend with each container (ax.containers): the
            # contained artists have already been added.
        else:
            artist = entry
            artists.append(artist)
    return Cursor(artists, **kwargs)
