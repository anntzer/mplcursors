from collections import ChainMap, Counter
from collections.abc import Iterable
from contextlib import suppress
import copy
from functools import partial
import sys
import weakref
from weakref import WeakKeyDictionary

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cbook import CallbackRegistry
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
    textcoords="offset points",
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
    dict(position=(-15, 15), ha="right", va="bottom"),
    dict(position=(15, 15), ha="left", va="bottom"),
    dict(position=(15, -15), ha="left", va="top"),
    dict(position=(-15, -15), ha="right", va="top"),
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
    """A string subclass solely for marking purposes.
    """


def _get_rounded_intersection_area(bbox_1, bbox_2):
    """Compute the intersection area between two bboxes, rounded to 8 digits.
    """
    # The rounding allows sorting areas without floating point issues.
    bbox = bbox_1.intersection(bbox_1, bbox_2)
    return (round(bbox.width * bbox.height / 1e-8) * 1e-8
            if bbox else 0)


def _is_alive(artist):
    """Check whether an artist is still present on an axes.
    """
    return bool(artist and artist.axes
                and (artist.container in artist.axes.containers
                     if isinstance(artist, _pick_info.ContainerArtist)
                     else artist.axes.findobj(lambda obj: obj is artist)))


def _reassigned_axes_event(event, ax):
    """Reassign *event* to *ax*.
    """
    event = copy.copy(event)
    event.xdata, event.ydata = (
        ax.transData.inverted().transform_point((event.x, event.y)))
    return event


class Cursor:
    """A cursor for selecting Matplotlib artists.

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
        """Construct a cursor.

        Parameters
        ----------

        artists : List[Artist]
            A list of artists that can be selected by this cursor.

        multiple : bool, optional
            Whether multiple artists can be "on" at the same time (defaults to
            False).

        highlight : bool, optional
            Whether to also highlight the selected artist.  If so,
            "highlighter" artists will be placed as the first item in the
            :attr:`extras` attribute of the `Selection`.

        hover : bool, optional
            Whether to select artists upon hovering instead of by clicking.
            (Hovering over an artist while a button is pressed will not trigger
            a selection; right clicking on an annotation will still remove it.)

        bindings : dict, optional
            A mapping of button and keybindings to actions.  Valid entries are:

            ================ ==================================================
            'select'         mouse button to select an artist
                             (default: 1)
            'deselect'       mouse button to deselect an artist
                             (default: 3)
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
            assign any binding to an action, set it to ``None``.

        annotation_kwargs : dict, optional
            Keyword argments passed to the `annotate
            <matplotlib.axes.Axes.annotate>` call.

        annotation_positions : List[dict], optional
            List of positions tried by the annotation positioning algorithm.

        highlight_kwargs : dict, optional
            Keyword arguments used to create a highlighted artist.
        """

        artists = list(artists)
        # Be careful with GC.
        self._artists = [weakref.ref(artist) for artist in artists]

        for artist in artists:
            type(self)._keep_alive.setdefault(artist, set()).add(self)

        self._multiple = multiple
        self._highlight = highlight

        self._visible = True
        self._enabled = True
        self._selections = []
        self._last_auto_position = None
        self._callbacks = CallbackRegistry()

        connect_pairs = [("key_press_event", self._on_key_press)]
        if hover:
            if multiple:
                raise ValueError("'hover' and 'multiple' are incompatible")
            connect_pairs += [
                ("motion_notify_event", self._hover_handler),
                ("button_press_event", self._hover_handler)]
        else:
            connect_pairs += [
                ("button_press_event", self._nonhover_handler)]
        self._disconnectors = [
            partial(canvas.mpl_disconnect, canvas.mpl_connect(*pair))
            for pair in connect_pairs
            for canvas in {artist.figure.canvas for artist in artists}]

        bindings = dict(ChainMap(bindings if bindings is not None else {},
                                 _default_bindings))
        unknown_bindings = set(bindings) - set(_default_bindings)
        if unknown_bindings:
            raise ValueError("Unknown binding(s): {}".format(
                ", ".join(sorted(unknown_bindings))))
        duplicate_bindings = [
            k for k, v in Counter(list(bindings.values())).items() if v > 1]
        if duplicate_bindings:
            raise ValueError("Duplicate binding(s): {}".format(
                ", ".join(sorted(map(str, duplicate_bindings)))))
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
        """The tuple of selectable artists.
        """
        # Work around matplotlib/matplotlib#6982: `cla()` does not clear
        # `.axes`.
        return tuple(filter(_is_alive, (ref() for ref in self._artists)))

    @property
    def enabled(self):
        """Whether clicks are registered for picking and unpicking events.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    @property
    def selections(self):
        """The tuple of current `Selection`\\s.
        """
        for sel in self._selections:
            if sel.annotation.axes is None:
                raise RuntimeError("Annotation unexpectedly removed; "
                                   "use 'cursor.remove_selection' instead")
        return tuple(self._selections)

    @property
    def visible(self):
        """Whether selections are visible by default.

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

    def add_selection(self, pi):
        """Create an annotation for a `Selection` and register it.

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
        figure = pi.artist.figure
        axes = pi.artist.axes
        if axes.get_renderer_cache() is None:
            figure.canvas.draw()  # Needed by draw_artist below anyways.
        renderer = pi.artist.axes.get_renderer_cache()
        ann = pi.artist.axes.annotate(
            _pick_info.get_ann_text(*pi), xy=pi.target,
            xytext=(np.nan, np.nan),
            ha=_MarkedStr("center"), va=_MarkedStr("center"),
            visible=self.visible,
            **self.annotation_kwargs)
        ann.draggable(use_blit=True)
        extras = []
        if self._highlight:
            hl = self.add_highlight(*pi)
            if hl:
                extras.append(hl)
        sel = pi._replace(annotation=ann, extras=extras)
        self._selections.append(sel)
        self._callbacks.process("add", sel)

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
            if isinstance(ann.get_ha(), _MarkedStr):
                ann.set_ha({-1: "right", 0: "center", 1: "left"}[
                    np.sign(np.nan_to_num(ann.xyann[0]))])
            if isinstance(ann.get_va(), _MarkedStr):
                ann.set_va({-1: "top", 0: "center", 1: "bottom"}[
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
            figure.draw_artist(ann)
            # Explicit argument needed on MacOSX backend.
            figure.canvas.blit(figure.bbox)
        # Removal comes after addition so that the fast blitting path works.
        if not self._multiple:
            for sel in self.selections[:-1]:
                self.remove_selection(sel)
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
            **ChainMap({"highlight_kwargs": self.highlight_kwargs}, kwargs))
        if hl:
            artist.axes.add_artist(hl)
            return hl

    def connect(self, event, func=None):
        """Connect a callback to a `Cursor` event; return the callback id.

        Two classes of event can be emitted, both with a `Selection` as single
        argument:

            - ``"add"`` when a `Selection` is added, and
            - ``"remove"`` when a `Selection` is removed.

        The callback registry relies on Matplotlib's implementation; in
        particular, only weak references are kept for bound methods.

        This method is can also be used as a decorator::

            @cursor.connect("add")
            def on_add(sel):
                ...

        Examples of callbacks::

            # Change the annotation text and alignment:
            lambda sel: sel.annotation.set(
                text=sel.artist.get_label(),  # or use e.g. sel.target.index
                ha="center", va="bottom")

            # Make label non-draggable:
            lambda sel: sel.draggable(False)
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
        """Remove a cursor.

        Remove all `Selection`\\s, disconnect all callbacks, and allow the
        cursor to be garbage collected.
        """
        for disconnectors in self._disconnectors:
            disconnectors()
        for sel in self.selections:
            self.remove_selection(sel)
        for s in type(self)._keep_alive.values():
            with suppress(KeyError):
                s.remove(self)

    def _nonhover_handler(self, event):
        if event.name == "button_press_event":
            if event.button == self.bindings["select"]:
                self._on_select_button_press(event)
            if event.button == self.bindings["deselect"]:
                self._on_deselect_button_press(event)

    def _hover_handler(self, event):
        if event.name == "motion_notify_event" and event.button is None:
            # Filter away events where the mouse is pressed, in particular to
            # avoid conflicts between hover and draggable.
            self._on_select_button_press(event)
        elif (event.name == "button_press_event"
              and event.button == self.bindings["deselect"]):
            # Still allow removing the annotation by right clicking.
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
        for sel in self.selections:
            ann = sel.annotation
            if event.canvas is not ann.figure.canvas:
                continue
            contained, _ = ann.contains(event)
            if contained:
                self.remove_selection(sel)

    def _on_key_press(self, event):
        if event.key == self.bindings["toggle_enabled"]:
            self.enabled = not self.enabled
        elif event.key == self.bindings["toggle_visible"]:
            self.visible = not self.visible
        try:
            sel = self.selections[-1]
        except IndexError:
            return
        for key in ["left", "right", "up", "down"]:
            if event.key == self.bindings[key]:
                self.remove_selection(sel)
                self.add_selection(_pick_info.move(*sel, key=key))
                break

    def remove_selection(self, sel):
        """Remove a `Selection`.
        """
        self._selections.remove(sel)
        # <artist>.figure will be unset so we save them first.
        figures = {artist.figure for artist in [sel.annotation] + sel.extras}
        # ValueError is raised if the artist has already been removed.
        with suppress(ValueError):
            sel.annotation.remove()
        for artist in sel.extras:
            with suppress(ValueError):
                artist.remove()
        self._callbacks.process("remove", sel)
        for figure in figures:
            figure.canvas.draw_idle()


def cursor(pickables=None, **kwargs):
    """Create a `Cursor` for a list of artists, containers, and axes.

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
                for artists in [entry.collections, entry.images, entry.lines,
                                entry.patches, entry.texts]:
                    yield from artists
                containers.extend(entry.containers)
            elif isinstance(entry, Container):
                containers.append(entry)
            else:
                yield entry

    containers = []
    artists = list(iter_unpack_axes(iter_unpack_figures(pickables)))
    for container in containers:
        contained = list(filter(None, container.get_children()))
        for artist in contained:
            with suppress(ValueError):
                artists.remove(artist)
        if contained:
            artists.append(_pick_info.ContainerArtist(container))

    return Cursor(artists, **kwargs)
