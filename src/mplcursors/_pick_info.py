# Unsupported Artist classes: subclasses of AxesImage, QuadMesh (upstream could
# have a `format_coord`-like method); PolyCollection (picking is not well
# defined).

from collections import namedtuple
from contextlib import suppress
import copy
import functools
import inspect
from inspect import Signature
from numbers import Integral
import re
import warnings
from weakref import WeakSet

from matplotlib.axes import Axes
from matplotlib.collections import (
    LineCollection, PatchCollection, PathCollection)
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.contour import ContourSet
from matplotlib.image import AxesImage, BboxImage
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetBox
from matplotlib.patches import Patch, PathPatch, Polygon, Rectangle
from matplotlib.quiver import Barbs, Quiver
from matplotlib.text import Text
from matplotlib.transforms import (
    Affine2D, Bbox, BboxTransformFrom, BboxTransformTo)
import numpy as np


PATCH_PICKRADIUS = 5  # FIXME Patches do not provide `pickradius`.


def _register_scatter():
    """
    Patch `PathCollection` and `scatter` to register their return values.

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
_register_scatter()


def _is_scatter(artist):
    return (isinstance(artist, PathCollection)
            and artist not in _nonscatter_pathcollections)


def _artist_in_container(container):
    return next(filter(None, container.get_children()))


class ContainerArtist:
    """Workaround to make containers behave more like artists."""

    def __init__(self, container):
        self.container = container  # Guaranteed to be nonempty.
        # We can't weakref the Container (which subclasses tuple), so
        # we instead create a reference cycle between the Container and
        # the ContainerArtist; as no one else strongly references the
        # ContainerArtist, it will get GC'd whenever the Container is.
        vars(container).setdefault(
            f"_{__class__.__name__}__keep_alive", []).append(self)

    def __str__(self):
        return f"<{type(self).__name__}({self.container})>"

    def __repr__(self):
        return f"<{type(self).__name__}({self.container!r})>"

    figure = property(lambda self: _artist_in_container(self.container).figure)
    axes = property(lambda self: _artist_in_container(self.container).axes)

    def get_visible(self):
        return True  # For lack of anything better.


Selection = namedtuple(
    "Selection", "artist target index dist annotation extras")
Selection.__doc__ = """
    A selection.

    Although this class is implemented as a namedtuple (to simplify the
    dispatching of `compute_pick`, `get_ann_text`, and `make_highlight`), only
    the field names should be considered stable API.  The number and order of
    fields are subject to change with no notice.
"""
# Override equality to identity: Selections should be considered immutable
# (with mutable fields though) and we don't want to trigger casts of array
# equality checks to booleans.  We don't need to override comparisons because
# artists are already non-comparable.
Selection.__eq__ = lambda self, other: self is other
Selection.__ne__ = lambda self, other: self is not other
Selection.artist.__doc__ = (
    "The selected artist.")
Selection.target.__doc__ = (
    "The point picked within the artist, in data coordinates.")
Selection.index.__doc__ = (
    "The index of the selected point, within the artist data.")
Selection.dist.__doc__ = (
    "The distance from the click to the target, in pixels.")
Selection.annotation.__doc__ = (
    "The instantiated `matplotlib.text.Annotation`.")
Selection.extras.__doc__ = (
    "An additional list of artists (e.g., highlighters) that will be cleared "
    "at the same time as the annotation.")


def _gen_warning_text(kind, tp):
    return "{} support for {} (MRO: {}) is missing.".format(
        kind, tp.__name__, ", ".join(cls.__name__ for cls in tp.__mro__))


@functools.singledispatch
def compute_pick(artist, event):
    """
    Find whether *artist* has been picked by *event*.

    If it has, return the appropriate `Selection`; otherwise return ``None``.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    warnings.warn(_gen_warning_text("Pick", type(artist)))


class Index:
    def __init__(self, i, x, y):
        self.int = i
        self.x = x
        self.y = y

    def __floor__(self):
        return self.int

    def __ceil__(self):
        return self.int if max(self.x, self.y) == 0 else self.int + 1

    # numpy<1.17 backcompat.
    floor = __floor__
    ceil = __ceil__

    def __format__(self, fmt):
        return f"{self.int}.(x={self.x:{fmt}}, y={self.y:{fmt}})"

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


def _compute_projection_pick(artist, path, xy):
    """
    Project *xy* on *path* to obtain a `Selection` for *artist*.

    *path* is first transformed to screen coordinates using the artist
    transform, and the target of the returned `Selection` is transformed
    back to data coordinates using the artist *axes* inverse transform.  The
    `Selection` `index` is returned as a float.  This function returns ``None``
    for degenerate inputs.

    The caller is responsible for converting the index to the proper class if
    needed.
    """
    transform = artist.get_transform().frozen()
    tpath = (path.cleaned(transform) if transform.is_affine
             # `cleaned` only handles affine transforms.
             else transform.transform_path(path).cleaned())
    # `cleaned` should return a path where the first element is `MOVETO`, the
    # following are `LINETO` or `CLOSEPOLY`, and the last one is `STOP`, i.e.
    #     codes = path.codes
    #     assert (codes[0], codes[-1]) == (path.MOVETO, path.STOP)
    #     assert np.isin(codes[1:-1], [path.LINETO, path.CLOSEPOLY]).all()
    vertices = tpath.vertices[:-1]
    codes = tpath.codes[:-1]
    mt_idxs, = (codes == tpath.MOVETO).nonzero()
    cp_idxs, = (codes == tpath.CLOSEPOLY).nonzero()
    vertices[cp_idxs] = vertices[mt_idxs[mt_idxs.searchsorted(cp_idxs) - 1]]
    # Unit vectors for each segment.
    us = vertices[1:] - vertices[:-1]
    ls = np.hypot(*us.T)
    with np.errstate(invalid="ignore"):
        # Results in 0/0 for repeated consecutive points.
        us /= ls[:, None]
    # Vectors from each vertex to the event.
    vs = xy - vertices[:-1]
    # Clipped dot products -- `einsum` cannot be done in place, `clip` can.
    # `clip` can trigger invalid comparisons if there are nan points.
    with np.errstate(invalid="ignore"):
        dot = np.clip(np.einsum("ij,ij->i", vs, us), 0, ls)
    # Projections.
    projs = vertices[:-1] + dot[:, None] * us
    ds = np.hypot(*(xy - projs).T)
    ds[mt_idxs[1:] - 1] = np.nan
    try:
        argmin = np.nanargmin(ds)
    except ValueError:  # Raised by nanargmin([nan]).
        return
    else:
        target = artist.axes.transData.inverted().transform(projs[argmin])
        index = ((argmin + dot[argmin] / ls[argmin])
                 / (path._interpolation_steps / tpath._interpolation_steps))
        return Selection(artist, target, index, ds[argmin], None, None)


def _untransform(orig_xy, screen_xy, ax):
    """
    Return data coordinates to place an annotation at screen coordinates
    *screen_xy* in axes *ax*.

    *orig_xy* are the "original" coordinates as stored by the artist; they are
    transformed to *screen_xy* by whatever transform the artist uses.  If the
    artist uses ``ax.transData``, just return *orig_xy*; else, apply
    ``ax.transData.inverse()`` to *screen_xy*.  (The first case is more
    accurate than always applying ``ax.transData.inverse()``.)
    """
    tr_xy = ax.transData.transform(orig_xy)
    return (
        orig_xy
        if ((tr_xy == screen_xy) | np.isnan(tr_xy) & np.isnan(screen_xy)).all()
        else ax.transData.inverted().transform(screen_xy))


@compute_pick.register(Line2D)
def _(artist, event):
    # No need to call `line.contains` as we're going to redo the work anyways
    # (also see matplotlib/matplotlib#6645, though that's fixed in mpl2.1).

    # Always work in screen coordinates, as this is how we need to compute
    # distances.  Note that the artist transform may be different from the axes
    # transform (e.g., for axvline).
    xy = event.x, event.y
    data_xy = artist.get_xydata()
    data_screen_xy = artist.get_transform().transform(data_xy)
    sels = []
    # If markers are visible, find the closest vertex.
    if artist.get_marker() not in ["None", "none", " ", "", None]:
        ds = np.hypot(*(xy - data_screen_xy).T)
        try:
            argmin = np.nanargmin(ds)
        except ValueError:  # Raised by nanargmin([nan]).
            pass
        else:
            target = _untransform(  # More precise than transforming back.
                data_xy[argmin], data_screen_xy[argmin], artist.axes)
            sels.append(
                Selection(artist, target, argmin, ds[argmin], None, None))
    # If lines are visible, find the closest projection.
    if (artist.get_linestyle() not in ["None", "none", " ", "", None]
            and len(artist.get_xydata()) > 1):
        sel = _compute_projection_pick(artist, artist.get_path(), xy)
        if sel is not None:
            sel = sel._replace(index={
                "_draw_lines": lambda _, index: index,
                "_draw_steps_pre": Index.pre_index,
                "_draw_steps_mid": Index.mid_index,
                "_draw_steps_post": Index.post_index}[
                    Line2D.drawStyles[artist.get_drawstyle()]](
                        len(data_xy), sel.index))
            sels.append(sel)
    sel = min(sels, key=lambda sel: sel.dist, default=None)
    return sel if sel and sel.dist < artist.get_pickradius() else None


@compute_pick.register(PathPatch)
@compute_pick.register(Polygon)
@compute_pick.register(Rectangle)
def _(artist, event):
    sel = _compute_projection_pick(
        artist, artist.get_path(), (event.x, event.y))
    if sel and sel.dist < PATCH_PICKRADIUS:
        return sel


@compute_pick.register(LineCollection)
@compute_pick.register(PatchCollection)
@compute_pick.register(PathCollection)
def _(artist, event):
    offsets = artist.get_offsets()
    paths = artist.get_paths()
    if _is_scatter(artist):
        # Use the C implementation to prune the list of segments -- but only
        # for scatter plots as that implementation is inconsistent with Line2D
        # for segment-like collections (matplotlib/matplotlib#17279).
        contains, info = artist.contains(event)
        if not contains:
            return
        inds = info["ind"]
        offsets = artist.get_offsets()[inds]
        offsets_screen = artist.get_offset_transform().transform(offsets)
        ds = np.hypot(*(offsets_screen - [event.x, event.y]).T)
        argmin = ds.argmin()
        target = _untransform(
            offsets[argmin], offsets_screen[argmin], artist.axes)
        return Selection(artist, target, inds[argmin], ds[argmin], None, None)
    elif len(paths) and len(offsets):
        # Note that this won't select implicitly closed paths.
        sels = [
            _compute_projection_pick(
                artist,
                Affine2D().translate(*offsets[ind % len(offsets)])
                .transform_path(paths[ind % len(paths)]),
                (event.x, event.y))
            for ind in range(max(len(offsets), len(paths)))]
        if not any(sels):
            return None
        idx = min(range(len(sels)),
                  key=lambda idx: sels[idx].dist if sels[idx] else np.inf)
        sel = sels[idx]
        if sel.dist >= artist.get_pickradius():
            return None
        return sel._replace(artist=artist, index=(idx, sel.index))


# This registration has no effect on mpl<3.8, where ContourSets are not artists
# and thus do not appear in the draw tree.
# Filled contours are picked identically to unfilled ones, in particular
# because Path.contains_point does not handle holes in paths correctly; thus we
# cannot determine which contour, among many nested ones, actually contains the
# a point between two layers.
@compute_pick.register(ContourSet)
def _(artist, event):
    return compute_pick.dispatch(LineCollection)(artist, event)


@compute_pick.register(AxesImage)
@compute_pick.register(BboxImage)
def _(artist, event):
    if type(artist) not in compute_pick.registry:
        # Skip and warn on subclasses (`NonUniformImage`, `PcolorImage`) as
        # they do not implement `contains` correctly.  Even if they did, they
        # would not support moving as we do not know where a given index maps
        # back physically.
        return compute_pick.dispatch(object)(artist, event)
    contains, _ = artist.contains(event)
    if not contains:
        return
    ns = np.asarray(artist.get_array().shape[:2])[::-1]  # (y, x) -> (x, y)
    xf, yf = BboxTransformFrom(
        artist.get_window_extent()).transform([event.x, event.y])
    if artist.origin == "upper":
        yf = 1 - yf
    idxs = np.minimum(((xf, yf) * ns).astype(int), ns - 1)[::-1]
    return Selection(artist, (event.xdata, event.ydata),
                     tuple(idxs), 0, None, None)


@compute_pick.register(Barbs)
@compute_pick.register(Quiver)
def _(artist, event):
    offsets = artist.get_offsets()
    offsets_screen = artist.get_offset_transform().transform(offsets)
    ds = np.hypot(*(offsets_screen - [event.x, event.y]).T)
    argmin = np.nanargmin(ds)
    if ds[argmin] < artist.get_pickradius():
        target = _untransform(
            offsets[argmin], offsets_screen[argmin], artist.axes)
        return Selection(artist, target, argmin, ds[argmin], None, None)
    else:
        return None


@compute_pick.register(Text)
def _(artist, event):
    return


@compute_pick.register(ContainerArtist)
def _(artist, event):
    return compute_pick(artist.container, event)


@compute_pick.register(BarContainer)
def _(container, event):
    try:
        (idx, patch), = {
            (idx, patch) for idx, patch in enumerate(container.patches)
            if patch.contains(event)[0]}
    except ValueError:
        return
    target = [event.xdata, event.ydata]
    if patch.sticky_edges.x:
        target[0], = (
            x for x in [patch.get_x(), patch.get_x() + patch.get_width()]
            if x not in patch.sticky_edges.x)
    if patch.sticky_edges.y:
        target[1], = (
            y for y in [patch.get_y(), patch.get_y() + patch.get_height()]
            if y not in patch.sticky_edges.y)
    return Selection(container, target, idx, 0, None, None)


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
        idx, _ = sel_err.index
        if data_line:
            target = data_line.get_xydata()[idx]
        else:  # We can't guess the original data in that case!
            return
        return Selection(container, target, idx, 0, None, None)
    else:
        return


@compute_pick.register(StemContainer)
def _(container, event):
    sel = compute_pick(container.markerline, event)
    if sel:
        return sel
    if not isinstance(container.stemlines, LineCollection):
        warnings.warn("Only stem plots created with use_line_collection=True "
                      "are supported.")
        return
    sel = compute_pick(container.stemlines, event)
    if sel:
        idx, _ = sel.index
        target = container.stemlines.get_segments()[idx][-1]
        return Selection(container, target, sel.index, 0, None, None)


@compute_pick.register(OffsetBox)
def _(artist, event):
    # Pass-through: actually picks a child artist.
    return min(
        filter(None, [compute_pick(child, event)
                      for child in artist.get_children()]),
        key=lambda sel: sel.dist, default=None)


@compute_pick.register(AnnotationBbox)
def _(artist, event):
    # Pass-through: actually picks a child artist.
    return compute_pick(artist.offsetbox, event)


def _call_with_selection(func=None, *, argname="artist"):
    """Decorator that passes a `Selection` built from the non-kwonly args."""

    if func is None:
        return functools.partial(_call_with_selection, argname=argname)

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
        ba.apply_defaults()
        sel = Selection(*ba.args, **ba.kwargs)
        return func(sel, **extra_kw)

    params = [*sel_sig.parameters.values(), *wrapped_kwonly_params]
    params[0] = params[0].replace(name=argname)
    wrapper.__signature__ = Signature(params)
    return wrapper


def _format_coord_unspaced(ax, pos):
    # This used to directly post-process the output of format_coord(), but got
    # switched to handling special projections separately due to the change in
    # formatting for rectilinear coordinates.
    if ax.name == "polar":
        return ax.format_coord(*pos).replace(", ", "\n")
    elif ax.name == "3d":  # Need to retrieve the actual line data coordinates.
        warnings.warn("3d coordinates not supported yet")
        return ""
    else:
        x, y = pos
        # In mpl<3.3 (before #16776) format_x/ydata included trailing
        # spaces, hence the rstrip() calls.  format_xdata/format_ydata do not
        # actually always return strs (see test_fixed_ticks_nonstr_labels),
        # hence the explicit cast.
        return (f"x={str(ax.format_xdata(x)).rstrip()}\n"
                f"y={str(ax.format_ydata(y)).rstrip()}")


@functools.singledispatch
@_call_with_selection
def get_ann_text(sel):
    """
    Compute an annotating text for an (unpacked) `Selection`.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    warnings.warn(_gen_warning_text("Annotation", type(sel.artist)))
    return ""


@get_ann_text.register(Line2D)
@get_ann_text.register(LineCollection)
@get_ann_text.register(PatchCollection)
@get_ann_text.register(PathCollection)
@get_ann_text.register(Patch)
@_call_with_selection
def _(sel):
    artist = sel.artist
    label = artist.get_label() or ""
    text = _format_coord_unspaced(sel.annotation.axes, sel.target)
    if (_is_scatter(artist)
            # Heuristic: is the artist colormapped?
            # Note that this doesn't handle size-mapping (which is more likely
            # to involve an arbitrary scaling).
            and artist.get_array() is not None
            and len(artist.get_array()) == len(artist.get_offsets())):
        value = artist.format_cursor_data(artist.get_array()[sel.index])
        text = f"{text}\n{value}"
    if re.match("[^_]", label):
        text = f"{label}\n{text}"
    return text


@get_ann_text.register(ContourSet)
@_call_with_selection
def _(sel):
    artist = sel.artist
    return "{}\n{}".format(
        _format_coord_unspaced(sel.annotation.axes, sel.target),
        artist.levels[sel.index[0] + artist.filled])


@get_ann_text.register(AxesImage)
@get_ann_text.register(BboxImage)
@_call_with_selection
def _(sel):
    artist = sel.artist
    text = _format_coord_unspaced(sel.annotation.axes, sel.target)
    cursor_text = artist.format_cursor_data(artist.get_array()[sel.index])
    return f"{text}\n{cursor_text}"


@get_ann_text.register(Barbs)
@_call_with_selection
def _(sel):
    artist = sel.artist
    text = "{}\n({!s}, {!s})".format(
        _format_coord_unspaced(sel.annotation.axes, sel.target),
        artist.u[sel.index], artist.v[sel.index])
    return text


@get_ann_text.register(Quiver)
@_call_with_selection
def _(sel):
    artist = sel.artist
    text = "{}\n({!s}, {!s})".format(
        _format_coord_unspaced(sel.annotation.axes, sel.target),
        artist.U[sel.index], artist.V[sel.index])
    return text


# NOTE: There is no get_ann_text(ContainerArtist) as the selection directly
# refers to the Container itself.


@get_ann_text.register(BarContainer)
@_call_with_selection(argname="container")
def _(sel):
    return _format_coord_unspaced(sel.annotation.axes, sel.target)


@get_ann_text.register(ErrorbarContainer)
@_call_with_selection(argname="container")
def _(sel):
    data_line, cap_lines, err_lcs = sel.artist
    ann_text = get_ann_text(*sel._replace(artist=data_line))
    if isinstance(sel.index, Integral):
        err_lcs = iter(err_lcs)
        for idx, (dir, has) in enumerate(
                zip("xy", [sel.artist.has_xerr, sel.artist.has_yerr])):
            if has:
                err = (next(err_lcs).get_paths()[sel.index].vertices
                       - data_line.get_xydata()[sel.index])[:, idx]
                err_s = [getattr(sel.annotation.axes, f"format_{dir}data")(e)
                         .rstrip()
                         for e in err]
                # We'd normally want to check err.sum() == 0, but that can run
                # into fp inaccuracies.
                signs = "+-\N{MINUS SIGN}"
                if len({s.lstrip(signs) for s in err_s}) == 1:
                    repl = rf"\1=$\2\\pm{err_s[1]}$\3"
                else:
                    # Replacing unicode minus by ascii minus don't change the
                    # rendering as the string is mathtext, but allows keeping
                    # the same tests across Matplotlib versions that use
                    # unicode minus and those that don't.
                    err_s = [("+" if not s.startswith(tuple(signs)) else "")
                             + s.replace("\N{MINUS SIGN}", "-")
                             for s in err_s]
                    repl = r"\1=$\2_{%s}^{%s}$\3" % tuple(err_s)
                ann_text = re.sub(f"({dir})=(.*)(\n?)", repl, ann_text)
    return ann_text


@get_ann_text.register(StemContainer)
@_call_with_selection(argname="container")
def _(sel):
    return get_ann_text(*sel._replace(artist=sel.artist.markerline))


@functools.singledispatch
@_call_with_selection
def move(sel, *, key):
    """
    Move an (unpacked) `Selection` following a keypress.

    This function is used to implement annotation displacement through the
    keyboard.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    return sel


def _move_within_points(sel, xys, *, key):
    # Avoid infinite loop in case everything became nan at some point.
    for _ in range(len(xys)):
        if key == "left":
            new_idx = int(np.ceil(sel.index) - 1) % len(xys)
        elif key == "right":
            new_idx = int(np.floor(sel.index) + 1) % len(xys)
        else:
            return sel
        sel = sel._replace(target=xys[new_idx], index=new_idx, dist=0)
        if np.isfinite(sel.target).all():
            return sel


@move.register(Line2D)
@_call_with_selection
def _(sel, *, key):
    data_xy = sel.artist.get_xydata()
    return _move_within_points(
        sel,
        _untransform(data_xy, sel.artist.get_transform().transform(data_xy),
                     sel.annotation.axes),
        key=key)


@move.register(PathCollection)
@_call_with_selection
def _(sel, *, key):
    if _is_scatter(sel.artist):
        offsets = sel.artist.get_offsets()
        return _move_within_points(
            sel,
            _untransform(
                offsets, sel.artist.get_offset_transform().transform(offsets),
                sel.annotation.axes),
            key=key)
    else:
        return sel


@move.register(AxesImage)
@move.register(BboxImage)
@_call_with_selection
def _(sel, *, key):
    ns = sel.artist.get_array().shape[:2]
    delta = np.array(
        {"left": [0, -1], "right": [0, +1], "down": [-1, 0], "up": [+1, 0]}[
            key])
    if sel.artist.origin == "upper":
        delta[0] *= -1
    idxs = (sel.index + delta) % ns
    yf, xf = (idxs + .5) / ns
    if sel.artist.origin == "upper":
        yf = 1 - yf
    if isinstance(sel.artist, AxesImage):  # Same as below, but more accurate.
        x0, x1, y0, y1 = sel.artist.get_extent()
        trf = BboxTransformTo(Bbox.from_extents([x0, y0, x1, y1]))
    elif isinstance(sel.artist, BboxImage):
        trf = (BboxTransformTo(sel.artist.get_window_extent())
               - sel.annotation.axes.transData)
    target = trf.transform([xf, yf])
    return sel._replace(target=target, index=tuple(idxs))


# NOTE: There is no move(ContainerArtist) as the selection directly
# refers to the Container itself.


@move.register(ErrorbarContainer)
@_call_with_selection(argname="container")
def _(sel, *, key):
    data_line, cap_lines, err_lcs = sel.artist
    return _move_within_points(sel, data_line.get_xydata(), key=key)


@functools.singledispatch
@_call_with_selection
def make_highlight(sel, *, highlight_kwargs):
    """
    Create a highlight for an (unpacked) `Selection`.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    warnings.warn(_gen_warning_text("Highlight", type(sel.artist)))


def _set_valid_props(artist, kwargs):
    """Set valid properties for the artist, dropping the others."""
    artist.set(**{k: kwargs[k] for k in kwargs if hasattr(artist, "set_" + k)})
    return artist


@make_highlight.register(Line2D)
@_call_with_selection
def _(sel, *, highlight_kwargs):
    hl = copy.copy(sel.artist)
    _set_valid_props(hl, highlight_kwargs)
    return hl


@make_highlight.register(LineCollection)
@make_highlight.register(PathCollection)
@_call_with_selection
def _(sel, *, highlight_kwargs):
    hl = copy.copy(sel.artist)
    if _is_scatter(sel.artist):
        offsets = hl.get_offsets()
        hl.set_offsets(np.where(
            np.arange(len(offsets))[:, None] == sel.index, offsets, np.nan))
    else:
        hl.set_paths([
            path.vertices if i == sel.index[0] else np.empty((0, 2))
            for i, path in enumerate(hl.get_paths())])
    _set_valid_props(hl, highlight_kwargs)
    return hl
