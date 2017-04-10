# Unsupported Artist classes: subclasses of AxesImage, QuadMesh (upstream could
# have a `format_coord`-like method); PolyCollection (picking is not well
# defined).

from collections import ChainMap, namedtuple
from contextlib import suppress
import copy
import functools
import inspect
from inspect import Signature
import itertools
from numbers import Integral
import re
import warnings
from weakref import WeakSet

from matplotlib import cbook
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.quiver import Barbs, Quiver
from matplotlib.text import Text
from matplotlib.transforms import Affine2D
import numpy as np


Integral.register(np.integer)  # Back-compatibility for numpy 1.7, 1.8.


def _register_scatter():
    """Patch `PathCollection` and `scatter` to register their return values.

    This registration allows us to distinguish `PathCollection`s created by
    `Axes.scatter`, which should use point-like picking, from others, which
    should use path-like picking.  The former is more common, so we store the
    latter instead; this also lets us guess the type better if this module is
    imported late.
    """

    @functools.wraps(PathCollection.__init__)
    def __init__(self, *args, **kwargs):
        _nonscatter_pathcollections.add(self)
        return __init__.__wrapped__(self, *args, **kwargs)
    PathCollection.__init__ = __init__

    @functools.wraps(Axes.scatter)
    def scatter(*args, **kwargs):
        paths = scatter.__wrapped__(*args, **kwargs)
        with suppress(KeyError):
            _nonscatter_pathcollections.remove(paths)
        return paths
    Axes.scatter = scatter


_nonscatter_pathcollections = WeakSet()
_is_scatter = lambda artist: (isinstance(artist, PathCollection)
                              and artist not in _nonscatter_pathcollections)
_register_scatter()


def _artist_in_container(container):
    return next(filter(None, container.get_children()))


class ContainerArtist:
    """Workaround to make containers behave more like artists.
    """

    def __init__(self, container):
        self.container = container  # Guaranteed to be nonempty.
        # We can't weakref the Container (which subclasses tuple), so
        # we instead create a reference cycle between the Container and
        # the ContainerArtist; as no one else strongly references the
        # ContainerArtist, it will get GC'd whenever the Container is.
        vars(container).setdefault(
            "_{}__keep_alive".format(__class__.__name__), []).append(self)

    def __str__(self):
        return "<{}({})>".format(type(self).__name__, self.container)

    def __getitem__(self, key):
        return self.container[key]

    figure = property(lambda self: _artist_in_container(self.container).figure)
    axes = property(lambda self: _artist_in_container(self.container).axes)


class AttrArray(np.ndarray):
    """An array subclass that can store additional attributes.
    """

    def __new__(cls, array):
        return np.asarray(array).view(cls)


Selection = namedtuple("Selection", "artist target dist annotation extras")
# Override equality to identity: Selections should be considered immutable
# (with mutable fields though) and we don't want to trigger casts of array
# equality checks to booleans.  We don't need to override comparisons because
# artists are already non-comparable.
Selection.__eq__ = lambda self, other: self is other
Selection.__ne__ = lambda self, other: self is not other
try:
    Selection.artist.__doc__ = (
        "The selected artist.")
    Selection.target.__doc__ = (
        "The point picked within the artist, in data coordinates.")
    Selection.dist.__doc__ = (
        "The distance from the click to the target, in pixels.")
    Selection.annotation.__doc__ = (
        "The instantiated `matplotlib.text.Annotation`.")
    Selection.extras.__doc__ = (
        "An additional list of artists (e.g., highlighters) that will be "
        "cleared at the same time as the annotation.")
except AttributeError:  # Read-only in Py3.4.
    pass


@functools.singledispatch
def compute_pick(artist, event):
    """Find whether *artist* has been picked by *event*.

    If it has, return the appropriate `Selection`; otherwise return ``None``.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    warnings.warn("Pick support for {} is missing.".format(type(artist)))


class Index:
    def __init__(self, i, x, y):
        self.int = i
        self.x = x
        self.y = y

    def floor(self):
        return self.int

    def ceil(self):
        return self.int if max(self.x, self.y) == 0 else self.int + 1

    def __format__(self, fmt):
        return "{0.int}.(x={0.x:{1}}, y={0.y:{1}})".format(self, fmt)

    def __str__(self):
        return format(self, "")

    @classmethod
    def pre_index(cls, n_pts, index):
        i, frac = divmod(index, 1)
        i, odd = divmod(i, 2)
        x, y = (0, frac) if not odd else (frac, 1)
        return cls(i, x, y)

    @classmethod
    def post_index(cls, n_pts, index):
        i, frac = divmod(index, 1)
        i, odd = divmod(i, 2)
        x, y = (frac, 0) if not odd else (1, frac)
        return cls(i, x, y)

    @classmethod
    def mid_index(cls, n_pts, index):
        i, frac = divmod(index, 1)
        if i == 0:
            frac = .5 + frac / 2
        elif i == 2 * n_pts - 2:  # One less line than points.
            frac = frac / 2
        quot, odd = divmod(i, 2)
        if not odd:
            if frac < .5:
                i = quot - 1
                x, y = frac + .5, 1
            else:
                i = quot
                x, y = frac - .5, 0
        else:
            i = quot
            x, y = .5, frac
        return cls(i, x, y)


def _check_clean_path(path):
    codes = path.codes
    assert (codes[0], codes[-1]) == (path.MOVETO, path.STOP)
    assert np.in1d(codes[1:-1], [path.LINETO, path.CLOSEPOLY]).all()


def _compute_projection_pick(artist, path_or_vertices, xy):
    """Project *xy* on *path_or_vertices* to obtain a `Selection` for *artist*.

    *path* is first transformed to screen coordinates using the artist
    transform, and the target of the returned `Selection` is transformed
    back to data coordinates using the artist *axes* inverse transform.  The
    `Selection` `index` is returned as a float.  This function returns ``None``
    for degenerate inputs.

    The caller is responsible for converting the index to the proper class if
    needed.
    """
    transform = artist.get_transform().frozen()
    if isinstance(path_or_vertices, np.ndarray):
        vertices = path_or_vertices
        vertices = (transform.transform(vertices) if transform.is_affine
                    # Geo transforms perform interpolation.
                    else transform.transform_path(MPath(vertices)).vertices)
    elif isinstance(path_or_vertices, MPath):
        path = path_or_vertices
        path = (path.cleaned(transform) if transform.is_affine
                # `cleaned` only handles affine transforms.
                else transform.transform_path(path).cleaned())
        # `cleaned` should return a path where the first element is
        # `MOVETO`, the following are `LINETO` or `CLOSEPOLY`, and the
        # last one is `STOP`.  In case of unexpected behavior, debug using
        # `_check_clean_path(path)`.
        vertices = path.vertices[:-1]
        codes = path.codes[:-1]
        vertices[codes == path.CLOSEPOLY] = vertices[0]
    else:
        raise TypeError("Unexpected input type")
    # Unit vectors for each segment.
    us = vertices[1:] - vertices[:-1]
    ls = np.hypot(*us.T)
    with np.errstate(invalid="ignore"):
        # Results in 0/0 for repeated consecutive points.
        us /= ls[:, None]
    # Vectors from each vertex to the event (overwritten below).
    vs = xy - vertices[:-1]
    # Clipped dot products -- `einsum` cannot be done in place, `clip` can.
    dot = np.clip(np.einsum("ij,ij->i", vs, us), 0, ls, out=vs[:, 0])
    # Projections.
    projs = vertices[:-1] + dot[:, None] * us
    ds = np.hypot(*(xy - projs).T, out=vs[:, 1])
    try:
        argmin = np.nanargmin(ds)
        dmin = ds[argmin]
    except (ValueError, IndexError):  # See above re: exceptions caught.
        return
    else:
        target = AttrArray(
            artist.axes.transData.inverted().transform_point(projs[argmin]))
        if transform.is_affine:
            target.index = argmin + dot[argmin] / ls[argmin]
        return Selection(artist, target, dmin, None, None)


@compute_pick.register(Line2D)
def _(artist, event):
    # No need to call `line.contains` because we're going to redo
    # the work anyways (and it was broken for step plots up to
    # matplotlib/matplotlib#6645).

    # Always work in screen coordinates, as this is how we need to compute
    # distances.  Note that the artist transform may be different from the axes
    # transform (e.g., for axvline).
    xy = event.x, event.y
    data_xy = artist.get_xydata()
    sels = []
    # If markers are visible, find the closest vertex.
    if artist.get_marker() not in ["None", "none", " ", "", None]:
        ds = np.hypot(*(xy - artist.get_transform().transform(data_xy)).T)
        try:
            argmin = np.nanargmin(ds)
            dmin = ds[argmin]
        except (ValueError, IndexError):
            # numpy 1.7.0's `nanargmin([nan])` returns nan, so
            # `ds[argmin]` raises IndexError.  In later versions of numpy,
            # `nanargmin([nan])` raises ValueError (the release notes for 1.8.0
            # are incorrect on this topic).
            pass
        else:
            # More precise than transforming back.
            target = AttrArray(artist.get_xydata()[argmin])
            target.index = argmin
            sels.append(Selection(artist, target, dmin, None, None))
    # If lines are visible, find the closest projection.
    if (artist.get_linestyle() not in ["None", "none", " ", "", None]
            and len(artist.get_xydata()) > 1):
        drawstyle = Line2D.drawStyles[artist.get_drawstyle()]
        drawstyle_conv = {
            "_draw_lines": lambda xs, ys: (xs, ys),
            "_draw_steps_pre": cbook.pts_to_prestep,
            "_draw_steps_mid": cbook.pts_to_midstep,
            "_draw_steps_post": cbook.pts_to_poststep}[drawstyle]
        sel = _compute_projection_pick(
            artist, np.asarray(drawstyle_conv(*data_xy.T)).T, xy)
        if sel is not None:
            if hasattr(sel.target, "index"):
                sel.target.index = {
                    "_draw_lines": lambda _, index: index,
                    "_draw_steps_pre": Index.pre_index,
                    "_draw_steps_mid": Index.mid_index,
                    "_draw_steps_post": Index.post_index}[drawstyle](
                        len(data_xy), sel.target.index)
            sels.append(sel)
    sel = min(sels, key=lambda sel: sel.dist, default=None)
    return sel if sel and sel.dist < artist.get_pickradius() else None


@compute_pick.register(PathPatch)
@compute_pick.register(Polygon)
@compute_pick.register(Rectangle)
def _(artist, event):
    sel = _compute_projection_pick(
        artist, artist.get_path(), (event.x, event.y))
    if sel and sel.dist < 5:  # FIXME Patches do not provide `pickradius`.
        return sel


@compute_pick.register(LineCollection)
def _(artist, event):
    contains, info = artist.contains(event)
    paths = artist.get_paths()
    sels = [_compute_projection_pick(artist, paths[ind], (event.x, event.y))
            for ind in info["ind"]]
    sel, index = min(
        ((sel, info["ind"][idx]) for idx, sel in enumerate(sels) if sel),
        key=lambda sel_idx: sel_idx[0].dist, default=(None, None))
    if sel:
        sel = sel._replace(artist=artist)
        sel.target.index = (index, getattr(sel.target, "index", None))
    return sel


@compute_pick.register(PathCollection)
def _(artist, event):
    # Use the C implementation to prune the list of segments.
    contains, info = artist.contains(event)
    if not contains:
        return
    offsets = artist.get_offsets()
    paths = artist.get_paths()
    if _is_scatter(artist):
        ax = artist.axes
        inds = info["ind"]
        offsets = offsets[inds]
        ds = np.hypot(*(ax.transData.transform(offsets)
                        - [event.x, event.y]).T)
        argmin = ds.argmin()
        target = AttrArray(offsets[argmin])
        target.index = inds[argmin]
        return Selection(artist, target, ds[argmin], None, None)
    else:
        # Note that this won't select implicitly closed paths.
        sels = [
            _compute_projection_pick(
                artist,
                Affine2D().translate(*offsets[ind % len(offsets)])
                .transform_path(paths[ind % len(paths)]),
                (event.x, event.y))
            for ind in info["ind"]]
        sel, index = min(((sel, idx) for idx, sel in enumerate(sels) if sel),
                         key=lambda sel_idx: sel_idx[0].dist,
                         default=(None, None))
        if sel:
            sel = sel._replace(artist=artist)
            sel.target.index = (index, getattr(sel.target, "index", None))
        return sel


@compute_pick.register(AxesImage)
def _(artist, event):
    if type(artist) != AxesImage:
        # Skip and warn on subclasses (`NonUniformImage`, `PcolorImage`) as
        # they do not implement `contains` correctly.  Even if they did, they
        # would not support moving as we do not know where a given index maps
        # back physically.
        return compute_pick.dispatch(object)(artist, event)
    contains, _ = artist.contains(event)
    if not contains:
        return
    return Selection(artist, (event.xdata, event.ydata), 0, None, None)


@compute_pick.register(Barbs)
@compute_pick.register(Quiver)
def _(artist, event):
    offsets = artist.get_offsets()
    ds = np.hypot(
        *(artist.axes.transData.transform(offsets) - [event.x, event.y]).T)
    argmin = np.nanargmin(ds)
    if ds[argmin] < artist.get_pickradius():
        target = AttrArray(offsets[argmin])
        target.index = argmin
        return Selection(artist, target, ds[argmin], None, None)
    else:
        return None


@compute_pick.register(Text)
def _(artist, event):
    return


@compute_pick.register(ContainerArtist)
def _(artist, event):
    sel = compute_pick(artist.container, event)
    if sel:
        sel = sel._replace(artist=artist)
    return sel


@compute_pick.register(BarContainer)
def _(container, event):
    try:
        (idx, patch), = {
            (idx, patch) for idx, patch in enumerate(container.patches)
            if patch.contains(event)[0]}
    except ValueError:
        return
    target = AttrArray([event.xdata, event.ydata])
    target.index = idx
    if patch.sticky_edges.x:
        target[0], = (
            x for x in [patch.get_x(), patch.get_x() + patch.get_width()]
            if x not in patch.sticky_edges.x)
    if patch.sticky_edges.y:
        target[1], = (
            y for y in [patch.get_y(), patch.get_y() + patch.get_height()]
            if y not in patch.sticky_edges.y)
    return Selection(None, target, 0, None, None)


@compute_pick.register(ErrorbarContainer)
def _(container, event):
    data_line, cap_lines, err_lcs = container
    sel_data = compute_pick(data_line, event) if data_line else None
    sel_err = min(
        filter(None, (compute_pick(err_lc, event) for err_lc in err_lcs)),
        key=lambda sel: sel.dist, default=None)
    if (sel_data and sel_data.dist < getattr(sel_err, "dist", np.inf)):
        return sel_data
    elif sel_err:
        idx, _ = sel_err.target.index
        if data_line:
            target = AttrArray(data_line.get_xydata()[idx])
            target.index = idx
        else:  # We can't guess the original data in that case!
            return
        return Selection(None, target, 0, None, None)
    else:
        return


@compute_pick.register(StemContainer)
def _(container, event):
    sel = compute_pick(container.markerline, event)
    if sel:
        return sel
    idx_sel = min(filter(lambda idx_sel: idx_sel[1] is not None,
                         ((idx, compute_pick(line, event))
                          for idx, line in enumerate(container.stemlines))),
                  key=lambda idx_sel: idx_sel[1].dist, default=None)
    if idx_sel:
        idx, _ = idx_sel
        target = AttrArray(container.stemlines[idx].get_xydata()[-1])
        target.index = idx
        return Selection(None, target, 0, None, None)


def _call_with_selection(func):
    """Decorator that passes a `Selection` built from the non-kwonly args.
    """
    wrapped_kwonly_params = [
        param for param in inspect.signature(func).parameters.values()
        if param.kind == param.KEYWORD_ONLY]
    sel_sig = inspect.signature(Selection)
    default_sel_sig = sel_sig.replace(
        parameters=[param.replace(default=None) if param.default is param.empty
                    else param
                    for param in sel_sig.parameters.values()])
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        extra_kw = {param.name: kwargs.pop(param.name)
                    for param in wrapped_kwonly_params if param.name in kwargs}
        ba = default_sel_sig.bind(*args, **kwargs)
        # apply_defaults
        ba.arguments = ChainMap(
            ba.arguments,
            {name: param.default
             for name, param in default_sel_sig.parameters.items()
             if param.default is not param.empty})
        sel = Selection(*ba.args, **ba.kwargs)
        return func(sel, **extra_kw)
    wrapper.__signature__ = Signature(
        list(sel_sig.parameters.values()) + wrapped_kwonly_params)
    return wrapper


def _format_coord_unspaced(ax, xy):
    # Un-space-pad, remove empty coordinates from the output of
    # `format_{x,y}data`, and rejoin with newlines.
    return "\n".join(
        line for line, empty in zip(
            re.split(",? +", ax.format_coord(*xy)),
            itertools.chain(["x=", "y=", "z="], itertools.repeat(None)))
        if line != empty).rstrip()


@functools.singledispatch
@_call_with_selection
def get_ann_text(sel):
    """Compute an annotating text for a `Selection` (passed unpacked).

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    warnings.warn(
        "Annotation support for {} is missing".format(type(sel.artist)))
    return ""


@get_ann_text.register(Line2D)
@get_ann_text.register(LineCollection)
@get_ann_text.register(PathCollection)
@get_ann_text.register(Patch)
@_call_with_selection
def _(sel):
    artist = sel.artist
    label = artist.get_label() or ""
    text = _format_coord_unspaced(artist.axes, sel.target)
    if re.match("[^_]", label):
        text = "{}\n{}".format(label, text)
    return text


@get_ann_text.register(AxesImage)
@_call_with_selection
def _(sel):
    artist = sel.artist
    text = _format_coord_unspaced(artist.axes, sel.target)
    event = namedtuple("event", "xdata ydata")(*sel.target)
    text += "\n[{}]".format(
        artist.format_cursor_data(artist.get_cursor_data(event)))
    return text


@get_ann_text.register(Barbs)
@get_ann_text.register(Quiver)
@_call_with_selection
def _(sel):
    artist = sel.artist
    if isinstance(artist, Barbs):
        u, v = artist.u, artist.v
    elif isinstance(artist, Quiver):
        u, v = artist.U, artist.V
    else:
        raise TypeError("Unexpected type")
    text = "{}\n{}".format(
        _format_coord_unspaced(artist.axes, sel.target),
        (u[sel.target.index], v[sel.target.index]))
    return text


@get_ann_text.register(ContainerArtist)
@_call_with_selection
def _(sel):
    return get_ann_text(*sel._replace(artist=sel.artist.container))


@get_ann_text.register(BarContainer)
@_call_with_selection
def _(sel):
    return _format_coord_unspaced(
        _artist_in_container(sel.artist).axes, sel.target)


@get_ann_text.register(ErrorbarContainer)
@_call_with_selection
def _(sel):
    data_line, cap_lines, err_lcs = sel.artist
    ann_text = get_ann_text(*sel._replace(artist=data_line))
    if isinstance(sel.target.index, Integral):
        err_lcs = iter(err_lcs)
        for idx, (dir, has) in enumerate(
                zip("xy", [sel.artist.has_xerr, sel.artist.has_yerr])):
            if has:
                err = (next(err_lcs).get_paths()[sel.target.index].vertices
                       - data_line.get_xydata()[sel.target.index])[:, idx]
                fmt = lambda e: (
                    getattr(_artist_in_container(sel.artist).axes,
                            "format_{}data".format(dir))(e).rstrip())
                if err.sum():
                    err_s = [("+" if not s.startswith(("+", "-")) else "") + s
                            for s in map(fmt, err)]
                    repl = r"\1=$\2_{{{}}}^{{{}}}$\3".format(*err_s)
                else:
                    repl = r"\1=$\2\pm{}$\3".format(fmt(err[1]))
                ann_text = re.sub("({})=(.*)(\n?)".format(dir), repl, ann_text)
    return ann_text


@get_ann_text.register(StemContainer)
@_call_with_selection
def _(sel):
    return get_ann_text(*sel._replace(artist=sel.artist.markerline))


@functools.singledispatch
@_call_with_selection
def move(sel, *, key):
    """Move a `Selection` (passed unpacked) following a keypress.

    This function is used to implement annotation displacement through the
    keyboard.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    return sel


def _move_within_points(sel, xys, *, key):
    # Avoid infinite loop in case everything became nan at some point.
    for _ in range(len(xys)):
        new_idx = (int(np.ceil(sel.target.index) - 1) if key == "left"
                   else int(np.floor(sel.target.index) + 1) if key == "right"
                   else sel.target.index) % len(xys)
        target = AttrArray(xys[new_idx])
        target.index = new_idx
        sel = sel._replace(target=target, dist=0)
        if np.isfinite(target).all():
            return sel


@move.register(Line2D)
@_call_with_selection
def _(sel, *, key):
    if not hasattr(sel.target, "index"):
        return sel
    return _move_within_points(sel, sel.artist.get_xydata(), key=key)


@move.register(PathCollection)
@_call_with_selection
def _(sel, *, key):
    if _is_scatter(sel.artist):
        return _move_within_points(sel, sel.artist.get_offsets(), key=key)
    else:
        return sel


@move.register(AxesImage)
@_call_with_selection
def _(sel, *, key):
    low, high = np.reshape(sel.artist.get_extent(), (2, 2)).T
    ns = np.asarray(sel.artist.get_array().shape)[::-1]  # (y, x) -> (x, y)
    idxs = ((sel.target - low) / (high - low) * ns).astype(int)
    idxs += {
        "left": [-1, 0], "right": [1, 0], "up": [0, 1], "down": [0, -1]}[key]
    idxs %= ns
    target = (idxs + .5) / ns * (high - low) + low
    return sel._replace(target=target)


@functools.singledispatch
@_call_with_selection
def make_highlight(sel, *, highlight_kwargs):
    """Create a highlight for a `Selection`.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    warnings.warn(
        "Highlight support for {} is missing".format(type(sel.artist)))


def _set_valid_props(artist, kwargs):
    """Set valid properties for the artist, dropping the others.
    """
    artist.set(**{k: kwargs[k] for k in kwargs if hasattr(artist, "set_" + k)})
    return artist


@make_highlight.register(Line2D)
@_call_with_selection
def _(sel, *, highlight_kwargs):
    hl = copy.copy(sel.artist)
    _set_valid_props(hl, highlight_kwargs)
    return hl


@make_highlight.register(PathCollection)
@_call_with_selection
def _(sel, *, highlight_kwargs):
    hl = copy.copy(sel.artist)
    offsets = hl.get_offsets()
    hl.set_offsets(np.where(
        np.arange(len(offsets))[:, None] == sel.target.index, offsets, np.nan))
    _set_valid_props(hl, highlight_kwargs)
    return hl
