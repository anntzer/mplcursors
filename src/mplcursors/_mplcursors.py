from collections.abc import Iterable
from contextlib import suppress
import copy
from enum import IntEnum
from functools import partial
import sys
import weakref
from weakref import WeakKeyDictionary, WeakSet

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.container import Container
from matplotlib.figure import Figure
import numpy as np

from . import _pick_info


_default_bindings = dict(
    select=1,
    deselect=3,
    left="shift+left",
    right="shift+right",
    up="shift+up",
    down="shift+down",
    toggle_enabled="e",
    toggle_visible="v",
)
_default_annotation_kwargs = dict(
    bbox=dict(
        boxstyle="round,pad=.5",
        fc="yellow",
        alpha=.5,
        ec="k",
    ),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3",
        shrinkB=0,
        ec="k",
    ),
)
_default_annotation_positions = [
    dict(position=(-15, 15), anncoords="offset points",
         horizontalalignment="right", verticalalignment="bottom"),
    dict(position=(15, 15), anncoords="offset points",
         horizontalalignment="left", verticalalignment="bottom"),
    dict(position=(15, -15), anncoords="offset points",
         horizontalalignment="left", verticalalignment="top"),
    dict(position=(-15, -15), anncoords="offset points",
         horizontalalignment="right", verticalalignment="top"),
]
_default_highlight_kwargs = dict(
    # Only the kwargs corresponding to properties of the artist will be passed.
    # Line2D.
    color="yellow",
    markeredgecolor="yellow",
    linewidth=3,
    markeredgewidth=3,
    # PathCollection.
    facecolor="yellow",
    edgecolor="yellow",
)


class _MarkedStr(str):
    """A string subclass solely for marking purposes."""


def _mouse_event_matches(event, spec):
    """
    Return whether a mouse event "matches" an event spec, which is either a
    single mouse button, or a mapping matched against ``vars(event)``, e.g.
    ``{"button": 1, "key": "control"}``.
    """
    if isinstance(spec, int):
        spec = {"button": spec}
    return all(getattr(event, k) == v for k, v in spec.items())


def _get_rounded_intersection_area(bbox_1, bbox_2):
    """Compute the intersection area between two bboxes rounded to 8 digits."""
    # The rounding allows sorting areas without floating point issues.
    bbox = bbox_1.intersection(bbox_1, bbox_2)
    return round(bbox.width * bbox.height, 8) if bbox else 0


def _iter_axes_subartists(ax):
    r"""Yield all child `Artist`\s (*not* `Container`\s) of *ax*."""
    yield from ax.collections
    yield from ax.images
    yield from ax.lines
    yield from ax.patches
    yield from ax.texts


def _is_alive(artist):
    """Check whether *artist* is still present on its parent axes."""
    return bool(
        artist
        and artist.axes
        # `cla()` clears `.axes` since matplotlib/matplotlib#24627 (3.7);
        # iterating over subartists can be very slow.
        and (getattr(mpl, "__version_info__", ()) >= (3, 7)
             or (artist.container in artist.axes.containers
                 if isinstance(artist, _pick_info.ContainerArtist) else
                 artist in _iter_axes_subartists(artist.axes))))


def _reassigned_axes_event(event, ax):
    """Reassign *event* to *ax*."""
    event = copy.copy(event)
    event.xdata, event.ydata = (
        ax.transData.inverted().transform((event.x, event.y)))
    return event


class HoverMode(IntEnum):
    NoHover, Persistent, Transient = range(3)


class Cursor:
    """
    A cursor for selecting Matplotlib artists.

    Attributes
    ----------
    bindings : dict
        See the *bindings* keyword argument to the constructor.
    annotation_kwargs : dict
        See the *annotation_kwargs* keyword argument to the constructor.
    annotation_positions : dict
        See the *annotation_positions* keyword argument to the constructor.
    highlight_kwargs : dict
        See the *highlight_kwargs* keyword argument to the constructor.
    """

    _keep_alive = WeakKeyDictionary()

    def __init__(self,
                 artists,
                 *,
                 multiple=False,
                 highlight=False,
                 hover=False,
                 bindings=None,
                 annotation_kwargs=None,
                 annotation_positions=None,
                 highlight_kwargs=None):
        """
        Construct a cursor.

        Parameters
        ----------

        artists : List[Artist]
            A list of artists that can be selected by this cursor.

        multiple : bool, default: False
            Whether multiple artists can be "on" at the same time.  If on,
            cursor dragging is disabled (so that one does not end up with many
            cursors on top of one another).

        highlight : bool, default: False
            Whether to also highlight the selected artist.  If so,
            "highlighter" artists will be placed as the first item in the
            :attr:`extras` attribute of the `Selection`.

        hover : `HoverMode`, default: False
            Whether to select artists upon hovering instead of by clicking.
            (Hovering over an artist while a button is pressed will not trigger
            a selection; right clicking on an annotation will still remove it.)
            Possible values are

            - False, alias `HoverMode.NoHover`: hovering is inactive.
            - True, alias `HoverMode.Persistent`: hovering is active;
              annotations remain in place even after the mouse moves away from
              the artist (until another artist is selected, if *multiple* is
              False).
            - 2, alias `HoverMode.Transient`: hovering is active; annotations
              are removed as soon as the mouse moves away from the artist.

        bindings : dict, optional
            A mapping of actions to button and keybindings.  Valid keys are:

            ================ ==================================================
            'select'         mouse button to select an artist
                             (default: :data:`.MouseButton.LEFT`)
            'deselect'       mouse button to deselect an artist
                             (default: :data:`.MouseButton.RIGHT`)
            'left'           move to the previous point in the selected path,
                             or to the left in the selected image
                             (default: shift+left)
            'right'          move to the next point in the selected path, or to
                             the right in the selected image
                             (default: shift+right)
            'up'             move up in the selected image
                             (default: shift+up)
            'down'           move down in the selected image
                             (default: shift+down)
            'toggle_enabled' toggle whether the cursor is active
                             (default: e)
            'toggle_visible' toggle default cursor visibility and apply it to
                             all cursors (default: v)
            ================ ==================================================

            Missing entries will be set to the defaults.  In order to not
            assign any binding to an action, set it to ``None``.  Modifier keys
            (or other event properties) can be set for mouse button bindings by
            passing them as e.g. ``{"button": 1, "key": "control"}``.

        annotation_kwargs : dict, default: {}
            Keyword argments passed to the `annotate
            <matplotlib.axes.Axes.annotate>` call.

        annotation_positions : List[dict], optional
            List of positions tried by the annotation positioning algorithm.
            The default is to try four positions, 15 points to the NW, NE, SE,
            and SW from the selected point; annotations that stay within the
            axes are preferred.

        highlight_kwargs : dict, default: {}
            Keyword arguments used to create a highlighted artist.
        """

        artists = [*artists]
        # Be careful with GC.
        self._artists = [weakref.ref(artist) for artist in artists]

        for artist in artists:
            type(self)._keep_alive.setdefault(artist, set()).add(self)

        self._multiple = multiple
        self._highlight = highlight

        self._visible = True
        self._enabled = True
        self._selections = []
        self._selection_stack = []
        self._last_auto_position = None
        self._callbacks = {"add": [], "remove": []}
        self._hover = hover

        self._suppressed_events = WeakSet()
        connect_pairs = [
            ("pick_event", self._on_pick),
            ("key_press_event", self._on_key_press),
        ]
        if hover:
            connect_pairs += [
                ("motion_notify_event", self._on_hover_motion_notify),
                ("button_press_event", self._on_hover_button_press),
            ]
        else:
            connect_pairs += [
                ("button_press_event", self._on_nonhover_button_press),
            ]
            if not self._multiple:
                connect_pairs.append(
                    ("motion_notify_event", self._on_nonhover_button_press))
        self._disconnectors = [
            partial(canvas.mpl_disconnect, canvas.mpl_connect(*pair))
            for pair in connect_pairs
            for canvas in {artist.figure.canvas for artist in artists}]

        bindings = {**_default_bindings,
                    **(bindings if bindings is not None else {})}
        unknown_bindings = {*bindings} - {*_default_bindings}
        if unknown_bindings:
            raise ValueError("Unknown binding(s): {}".format(
                ", ".join(sorted(unknown_bindings))))
        bindings_items = list(bindings.items())
        for i in range(len(bindings)):
            action, key = bindings_items[i]
            for j in range(i):
                other_action, other_key = bindings_items[j]
                if key == other_key and key is not None:
                    raise ValueError(
                        f"Duplicate bindings: {key} is used for "
                        f"{other_action} and for {action}")
        self.bindings = bindings

        self.annotation_kwargs = (
            annotation_kwargs if annotation_kwargs is not None
            else copy.deepcopy(_default_annotation_kwargs))
        self.annotation_positions = (
            annotation_positions if annotation_positions is not None
            else copy.deepcopy(_default_annotation_positions))
        self.highlight_kwargs = (
            highlight_kwargs if highlight_kwargs is not None
            else copy.deepcopy(_default_highlight_kwargs))

    @property
    def artists(self):
        """The tuple of selectable artists."""
        return tuple(filter(_is_alive, (ref() for ref in self._artists)))

    @property
    def enabled(self):
        """Whether clicks are registered for picking and unpicking events."""
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    @property
    def selections(self):
        r"""The tuple of current `Selection`\s."""
        for sel in self._selections:
            if sel.annotation.axes is None:
                raise RuntimeError("Annotation unexpectedly removed; "
                                   "use 'cursor.remove_selection' instead")
        return tuple(self._selections)

    @property
    def visible(self):
        """
        Whether selections are visible by default.

        Setting this property also updates the visibility status of current
        selections.
        """
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value
        for sel in self.selections:
            sel.annotation.set_visible(value)
            sel.annotation.figure.canvas.draw_idle()

    def _get_figure(self, aoc):
        """Return the parent figure of artist-or-container *aoc*."""
        if isinstance(aoc, Container):
            try:
                ca, = {artist for artist in (ref() for ref in self._artists)
                       if isinstance(artist, _pick_info.ContainerArtist)
                          and artist.container is aoc}
            except ValueError:
                raise ValueError(f"Cannot find parent figure of {aoc}")
            return ca.figure
        else:
            return aoc.figure

    def _get_axes(self, aoc):
        """Return the parent axes of artist-or-container *aoc*."""
        if isinstance(aoc, Container):
            try:
                ca, = {artist for artist in (ref() for ref in self._artists)
                       if isinstance(artist, _pick_info.ContainerArtist)
                          and artist.container is aoc}
            except ValueError:
                raise ValueError(f"Cannot find parent axes of {aoc}")
            return ca.axes
        else:
            return aoc.axes

    def add_selection(self, pi):
        """
        Create an annotation for a `Selection` and register it.

        Returns a new `Selection`, that has been registered by the `Cursor`,
        with the added annotation set in the :attr:`annotation` field and, if
        applicable, the highlighting artist in the :attr:`extras` field.

        Emits the ``"add"`` event with the new `Selection` as argument.  When
        the event is emitted, the position of the annotation is temporarily
        set to ``(nan, nan)``; if this position is not explicitly set by a
        callback, then a suitable position will be automatically computed.

        Likewise, if the text alignment is not explicitly set but the position
        is, then a suitable alignment will be automatically computed.
        """
        # pi: "pick_info", i.e. an incomplete selection.
        # Pre-fetch the figure and axes, as callbacks may actually unset them.
        figure = self._get_figure(pi.artist)
        axes = self._get_axes(pi.artist)
        get_cached_renderer = (
            figure.canvas.get_renderer
            if hasattr(figure.canvas, "get_renderer")
            else axes.get_renderer_cache)  # mpl<3.6.
        renderer = get_cached_renderer()
        if renderer is None:
            figure.canvas.draw()  # Needed below anyways.
            renderer = get_cached_renderer()
        ann = axes.annotate(
            _pick_info.get_ann_text(*pi), xy=pi.target,
            xytext=(np.nan, np.nan),
            horizontalalignment=_MarkedStr("center"),
            verticalalignment=_MarkedStr("center"),
            visible=self.visible,
            zorder=np.inf,
            **self.annotation_kwargs)
        # Move the Annotation's ownership from the Axes to the Figure, so that
        # it gets drawn even above twinned axes.  But ann.axes must stay set,
        # so that e.g. unit converters get correctly applied.
        ann.remove()
        ann.axes = axes
        figure.add_artist(ann)
        ann.draggable(use_blit=not self._multiple)
        extras = []
        if self._highlight:
            hl = self.add_highlight(*pi)
            if hl:
                extras.append(hl)
        sel = pi._replace(annotation=ann, extras=extras)
        self._selections.append(sel)
        self._selection_stack.append(sel)
        for cb in self._callbacks["add"]:
            cb(sel)

        # Check that `ann.axes` is still set, as callbacks may have removed the
        # annotation.
        if ann.axes and ann.xyann == (np.nan, np.nan):
            fig_bbox = figure.get_window_extent()
            ax_bbox = axes.get_window_extent()
            overlaps = []
            for idx, annotation_position in enumerate(
                    self.annotation_positions):
                ann.set(**annotation_position)
                # Work around matplotlib/matplotlib#7614: position update is
                # missing.
                ann.update_positions(renderer)
                bbox = ann.get_window_extent(renderer)
                overlaps.append(
                    (_get_rounded_intersection_area(fig_bbox, bbox),
                     _get_rounded_intersection_area(ax_bbox, bbox),
                     # Avoid needlessly jumping around by breaking ties using
                     # the last used position as default.
                     idx == self._last_auto_position))
            auto_position = max(range(len(overlaps)), key=overlaps.__getitem__)
            ann.set(**self.annotation_positions[auto_position])
            self._last_auto_position = auto_position
        else:
            if isinstance(ann.get_horizontalalignment(), _MarkedStr):
                ann.set_horizontalalignment(
                    {-1: "right", 0: "center", 1: "left"}[
                        np.sign(np.nan_to_num(ann.xyann[0]))])
            if isinstance(ann.get_verticalalignment(), _MarkedStr):
                ann.set_verticalalignment(
                    {-1: "top", 0: "center", 1: "bottom"}[
                        np.sign(np.nan_to_num(ann.xyann[1]))])

        if (extras
                or len(self.selections) > 1 and not self._multiple
                or not figure.canvas.supports_blit):
            # Either:
            #  - there may be more things to draw, or
            #  - annotation removal will make a full redraw necessary, or
            #  - blitting is not (yet) supported.
            figure.canvas.draw_idle()
        elif ann.axes:
            # Fast path, only needed if the annotation has not been immediately
            # removed.
            ann.draw(renderer)
            figure.canvas.blit()
        # Removal comes after addition so that the fast blitting path works.
        if not self._multiple:
            for sel in self.selections[:-1]:
                self.remove_selection(sel)
        return sel

    def add_highlight(self, artist, *args, **kwargs):
        """
        Create, add, and return a highlighting artist.

        This method is should be called with an "unpacked" `Selection`,
        possibly with some fields set to None.

        It is up to the caller to register the artist with the proper
        `Selection` (by calling ``sel.extras.append`` on the result of this
        method) in order to ensure cleanup upon deselection.
        """
        hl = _pick_info.make_highlight(
            artist, *args,
            **{"highlight_kwargs": self.highlight_kwargs, **kwargs})
        if hl:
            artist.axes.add_artist(hl)
            return hl

    def connect(self, event, func=None):
        """
        Connect a callback to a `Cursor` event; return the callback.

        Two events can be connected to:

        - callbacks connected to the ``"add"`` event are called when a
          `Selection` is added, with that selection as only argument;
        - callbacks connected to the ``"remove"`` event are called when a
          `Selection` is removed, with that selection as only argument.

        This method can also be used as a decorator::

            @cursor.connect("add")
            def on_add(sel):
                ...

        Examples of callbacks::

            # Change the annotation text and alignment:
            lambda sel: sel.annotation.set(
                text=sel.artist.get_label(),  # or use e.g. sel.index
                ha="center", va="bottom")

            # Make label non-draggable:
            lambda sel: sel.draggable(False)

        Note that when a single event causes both the removal of an "old"
        selection and the addition of a "new" one (typically, clicking on an
        artist when another one is selected, or hovering -- both assuming that
        ``multiple=False``), the "add" callback is called *first*.  This allows
        it, in particular, to "cancel" the addition (by immediately removing
        the "new" selection) and thus avoid removing the "old" selection.
        However, this call order may change in a future release.
        """
        if event not in self._callbacks:
            raise ValueError(f"{event!r} is not a valid cursor event")
        if func is None:
            return partial(self.connect, event)
        self._callbacks[event].append(func)
        return func

    def disconnect(self, event, cb):
        """
        Disconnect a previously connected callback.

        If a callback is connected multiple times, only one connection is
        removed.
        """
        try:
            self._callbacks[event].remove(cb)
        except KeyError:
            raise ValueError(f"{event!r} is not a valid cursor event")
        except ValueError:
            raise ValueError(f"Callback {cb} is not registered to {event}")

    def remove(self):
        """
        Remove a cursor.

        Remove all `Selection`\\s, disconnect all callbacks, and allow the
        cursor to be garbage collected.
        """
        for disconnector in self._disconnectors:
            disconnector()
        for sel in self.selections:
            self.remove_selection(sel)
        for s in type(self)._keep_alive.values():
            with suppress(KeyError):
                s.remove(self)

    def _on_pick(self, event):
        # Avoid creating a new annotation when dragging a preexisting
        # annotation (if multiple = True).  To do so, rely on the fact that
        # pick_events (which are used to implement dragging) trigger first (via
        # Figure's button_press_event, which is registered first); when one of
        # our annotations is picked, registed the corresponding mouse event as
        # "suppressed".  This can be done via a WeakSet as Matplotlib will keep
        # the event alive while being propagated through the callbacks.
        # Additionally, also rely on this mechanism to update the "current"
        # selection.
        for sel in self._selections:
            if event.artist is sel.annotation:
                self._suppressed_events.add(event.mouseevent)
                self._selection_stack.remove(sel)
                self._selection_stack.append(sel)
                break

    def _on_nonhover_button_press(self, event):
        if _mouse_event_matches(event, self.bindings["select"]):
            self._on_select_event(event)
        if _mouse_event_matches(event, self.bindings["deselect"]):
            self._on_deselect_event(event)

    def _on_hover_motion_notify(self, event):
        if event.button is None:
            # Filter away events where the mouse is pressed, in particular to
            # avoid conflicts between hover and draggable.
            self._on_select_event(event)

    def _on_hover_button_press(self, event):
        if _mouse_event_matches(event, self.bindings["deselect"]):
            # Still allow removing the annotation by right clicking.
            self._on_deselect_event(event)

    def _filter_mouse_event(self, event):
        # Accept the event iff we are enabled, and either
        # - no other widget is active, and this is not the second click of a
        #   double click (to prevent double selection), or
        # - another widget is active, and this is a double click (to bypass
        #   the widget lock), or
        # - hovering is active (in which case this is a motion_notify_event
        #   anyways).
        return (self.enabled
                and (event.canvas.widgetlock.locked() == event.dblclick
                     or self._hover))

    def _on_select_event(self, event):
        if (not self._filter_mouse_event(event)
                # See _on_pick.  (We only suppress selects, not deselects.)
                or event in self._suppressed_events):
            return
        # Work around lack of support for twinned axes.
        per_axes_event = {ax: _reassigned_axes_event(event, ax)
                          for ax in {artist.axes for artist in self.artists}}
        pis = []
        for artist in self.artists:
            if (artist.axes is None  # Removed or figure-level artist.
                    or event.canvas is not artist.figure.canvas
                    or not artist.get_visible()
                    or not artist.axes.contains(event)[0]):  # Cropped by axes.
                continue
            pi = _pick_info.compute_pick(artist, per_axes_event[artist.axes])
            if pi:
                pis.append(pi)
        # The any() check avoids picking an already selected artist at the same
        # point, as likely the user is just dragging it.  We check this here
        # rather than not adding the pick_info to pis at all, because in
        # transient hover mode, selections should be cleared out only when no
        # candidate picks (including such duplicates) exist at all.
        pi = min((pi for pi in pis
                  if not any((pi.artist, tuple(pi.target))
                             == (other.artist, tuple(other.target))
                             for other in self._selections)),
                 key=lambda pi: pi.dist, default=None)
        if pi:
            self.add_selection(pi)
        elif not pis and self._hover == HoverMode.Transient:
            for sel in self.selections:
                if event.canvas is sel.annotation.figure.canvas:
                    self.remove_selection(sel)

    def _on_deselect_event(self, event):
        if not self._filter_mouse_event(event):
            return
        for sel in self.selections[::-1]:  # LIFO.
            ann = sel.annotation
            if event.canvas is not ann.figure.canvas:
                continue
            if ann.contains(event)[0]:
                self.remove_selection(sel)
                break
        else:
            if self._highlight:
                for sel in self.selections[::-1]:
                    if any(extra.contains(event)[0] for extra in sel.extras):
                        self.remove_selection(sel)
                        break

    def _on_key_press(self, event):
        if event.key == self.bindings["toggle_enabled"]:
            self.enabled = not self.enabled
        elif event.key == self.bindings["toggle_visible"]:
            self.visible = not self.visible
        if not self._selections or not self.enabled:
            return
        sel = self._selection_stack[-1]
        for key in ["left", "right", "up", "down"]:
            if event.key == self.bindings[key]:
                self.remove_selection(sel)
                self.add_selection(_pick_info.move(*sel, key=key))
                break

    def remove_selection(self, sel):
        """Remove a `Selection`."""
        self._selections.remove(sel)
        self._selection_stack.remove(sel)
        # <artist>.figure will be unset so we save them first.
        figures = {artist.figure for artist in [sel.annotation] + sel.extras}
        # ValueError is raised if the artist has already been removed.
        with suppress(ValueError):
            sel.annotation.remove()
        for artist in sel.extras:
            with suppress(ValueError):
                artist.remove()
        for cb in self._callbacks["remove"]:
            cb(sel)
        for figure in figures:
            figure.canvas.draw_idle()


def cursor(pickables=None, **kwargs):
    """
    Create a `Cursor` for a list of artists, containers, and axes.

    Parameters
    ----------

    pickables : Optional[List[Union[Artist, Container, Axes, Figure]]]
        All artists and containers in the list or on any of the axes or
        figures passed in the list are selectable by the constructed `Cursor`.
        Defaults to all artists and containers on any of the figures that
        :mod:`~matplotlib.pyplot` is tracking.  Note that the latter will only
        work when relying on pyplot, not when figures are directly instantiated
        (e.g., when manually embedding Matplotlib in a GUI toolkit).

    **kwargs
        Keyword arguments are passed to the `Cursor` constructor.
    """

    # Explicit check to avoid a confusing
    # "TypeError: Cursor.__init__() got multiple values for argument 'artists'"
    if "artists" in kwargs:
        raise TypeError(
            "cursor() got an unexpected keyword argument 'artists'")

    if pickables is None:
        # Do not import pyplot ourselves to avoid forcing the backend.
        plt = sys.modules.get("matplotlib.pyplot")
        pickables = [
            plt.figure(num) for num in plt.get_fignums()] if plt else []
    elif (isinstance(pickables, Container)
          or not isinstance(pickables, Iterable)):
        pickables = [pickables]

    def iter_unpack_figures(pickables):
        for entry in pickables:
            if isinstance(entry, Figure):
                yield from entry.axes
            else:
                yield entry

    def iter_unpack_axes(pickables):
        for entry in pickables:
            if isinstance(entry, Axes):
                yield from _iter_axes_subartists(entry)
                containers.extend(entry.containers)
            elif isinstance(entry, Container):
                containers.append(entry)
            else:
                yield entry

    containers = []
    artists = [*iter_unpack_axes(iter_unpack_figures(pickables))]
    for container in containers:
        contained = [*filter(None, container.get_children())]
        for artist in contained:
            with suppress(ValueError):
                artists.remove(artist)
        if contained:
            artists.append(_pick_info.ContainerArtist(container))

    return Cursor(artists, **kwargs)
