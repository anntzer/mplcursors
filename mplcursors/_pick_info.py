# Unsupported Artist classes: subclasses of AxesImage, QuadMesh (upstream could
# have a `format_coord`-like method); PolyCollection (picking is not well
# defined).

from collections import ChainMap, namedtuple
import copy
import functools
import inspect
from inspect import Signature, Parameter
import re
from unittest.mock import NonCallableMock
import warnings

from matplotlib import cbook
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.quiver import Barbs, Quiver
from matplotlib.text import Text
import numpy as np


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
    """Find whether ``artist`` has been picked by ``event``.

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
    def pre_index(cls, n_pts, raw_index, frac):
        i, odd = divmod(raw_index, 2)
        x, y = (0, frac) if not odd else (frac, 1)
        return cls(i, x, y)

    @classmethod
    def post_index(cls, n_pts, raw_index, frac):
        i, odd = divmod(raw_index, 2)
        x, y = (frac, 0) if not odd else (1, frac)
        return cls(i, x, y)

    @classmethod
    def mid_index(cls, n_pts, raw_index, frac):
        if raw_index == 0:
            frac = .5 + frac / 2
        elif raw_index == n_pts - 2:  # One less line than points.
            frac = frac / 2
        quot, odd = divmod(raw_index, 2)
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


def _compute_projection_pick(xy, vertices):
    """Find the closest projection of `xy` on the `vertices` broken line.

    This function takes inputs in screen coordinates and returns a `Selection`
    in screen coordinates as well.  The `Selection` `index` is returned as a
    float.  This function returns None for degenerate inputs.

    The caller is responsible for setting the `Selection` `artist`, converting
    the target back to data coordinates, and converting the index to the proper
    class, or removing it if the transform is not linear (as it is likely
    invalid in that case).
    """
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
        target = AttrArray(projs[argmin])
        target.index = argmin + dot[argmin] / ls[argmin]
        return Selection(None, target, dmin, None, None)


@compute_pick.register(Line2D)
def _(artist, event):
    # No need to call `line.contains` because we're going to redo
    # the work anyways (and it was broken for step plots up to
    # matplotlib/matplotlib#6645).

    # Always work in screen coordinates, as this is how we need to compute
    # distances.  Note that the artist transform may be different from the axes
    # transform (e.g., for axvline).
    xy = event.x, event.y
    sels = []
    # If markers are visible, find the closest vertex.
    if artist.get_marker() not in ["None", "none", " ", "", None]:
        vertices = artist.get_transform().transform(artist.get_xydata())
        ds = np.hypot(*(xy - vertices).T)
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
        data_xys = np.asarray(drawstyle_conv(*artist.get_xydata().T)).T
        transform = artist.get_transform()
        vertices = (
            transform.transform(data_xys) if transform.is_affine
            # Only construct Paths if we need to follow a curved projection.
            else transform.transform_path(MPath(data_xys)).vertices)
        sel = _compute_projection_pick(xy, vertices)
        if sel is not None:
            sel = sel._replace(artist=artist)
            sel.target[:] = artist.axes.transData.inverted().transform_point(
                sel.target)
            if transform.is_affine:
                sel.target.index = {
                    "_draw_lines": lambda _, x, y: x + y,
                    "_draw_steps_pre": Index.pre_index,
                    "_draw_steps_mid": Index.mid_index,
                    "_draw_steps_post": Index.post_index}[drawstyle](
                        len(vertices), *divmod(sel.target.index, 1))
            else:
                del sel.target.index
            sels.append(sel)
    sel = min(sels, key=lambda sel: sel.dist, default=None)
    return sel if sel and sel.dist < artist.get_pickradius() else None


def _check_clean_path(path):
    codes = path.codes
    assert (codes[0], codes[-1]) == (path.MOVETO, path.STOP)
    assert np.in1d(codes[1:-1], [path.LINETO, path.CLOSEPOLY]).all()


@compute_pick.register(PathPatch)
@compute_pick.register(Polygon)
@compute_pick.register(Rectangle)
def _(artist, event):
    transform = artist.get_transform()
    path = (artist.get_path().cleaned(transform) if transform.is_affine
            # `cleaned` only handles affine transforms.
            else transform.transform_path(artist.get_path()).cleaned())
    # `cleaned` should return a path where the first element is `MOVETO`, the
    # following are `LINETO` or `CLOSEPOLY`, and the last one is `STOP`.  In
    # case of unexpected behavior, debug using `_check_clean_path(path)`.
    vertices = path.vertices[:-1]
    codes = path.codes[:-1]
    vertices[codes == path.CLOSEPOLY] = vertices[0]
    sel = _compute_projection_pick((event.x, event.y), vertices)
    sel.target[:] = artist.axes.transData.inverted().transform_point(
        sel.target)
    if sel and sel.dist < 5:  # FIXME Patches do not provide `pickradius`.
        return sel._replace(artist=artist)


@compute_pick.register(LineCollection)
def _(artist, event):
    # (Faster that iterating over all segments ourselves.)
    contains, info = artist.contains(event)
    mock = NonCallableMock(wraps=artist)
    mock.get_xydata = lambda: segment
    mock.get_drawstyle = lambda: "default"
    mock.get_marker = lambda: "none"
    segments = artist.get_segments()
    sels = []
    for index in info["ind"]:
        segment = segments[index]
        sels.append(compute_pick.dispatch(Line2D)(mock, event))
    sel, index = min(((sel, idx) for idx, sel in enumerate(sels) if sel),
                     key=lambda sel_idx: sel_idx[0].dist, default=(None, None))
    if sel:
        sel = sel._replace(artist=artist)
        sel.target.index = (index, getattr(sel.target, "index", None))
    return sel


@compute_pick.register(PathCollection)
def _(artist, event):
    contains, info = artist.contains(event)
    if not contains:
        return
    # Snapping, really only works for scatter plots (for example,
    # `PathCollection`s created through `matplotlib.tri` are unsupported).
    if len(artist.get_paths()) != 1:
        warnings.warn("Only PathCollections created through `plt.scatter` are "
                      "supported.")
        return
    ax = artist.axes
    idxs = info["ind"]
    offsets = artist.get_offsets()[idxs]
    ds = np.hypot(*(ax.transData.transform(offsets) - [event.x, event.y]).T)
    argmin = ds.argmin()
    target = AttrArray(offsets[argmin])
    target.index = idxs[argmin]
    return Selection(artist, target, ds[argmin], None, None)


@compute_pick.register(AxesImage)
def _(artist, event):
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
        [Parameter("args", Parameter.VAR_POSITIONAL)]
        + wrapped_kwonly_params
        + [Parameter("kwargs", Parameter.VAR_KEYWORD)])
    return wrapper


def _format_coord_unspaced(ax, x, y):
    # Un-space-pad the output of `format_{x,y}data`.
    return re.sub("[ ,] +", "\n", ax.format_coord(x, y)).strip()


@functools.singledispatch
@_call_with_selection
def get_ann_text(sel):
    """Compute an annotating text for a `Selection`.

    The `Selection` is built from ``*args, **kwargs``.

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
    text = _format_coord_unspaced(artist.axes, *sel.target)
    if re.match("[^_]", label):
        text = "{}\n{}".format(label, text)
    return text


@get_ann_text.register(AxesImage)
@_call_with_selection
def _(sel):
    artist = sel.artist
    text = _format_coord_unspaced(artist.axes, *sel.target)
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
        _format_coord_unspaced(artist.axes, *sel.target),
        (u[sel.target.index], v[sel.target.index]))
    return text


@functools.singledispatch
@_call_with_selection
def move(sel, *, key):
    """Move a `Selection` following a keypress.

    The `Selection` is built from ``*args, **kwargs``.

    This function is used to implement annotation displacement through the
    keyboard.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    return sel


@move.register(Line2D)
@_call_with_selection
def _(sel, *, key):
    if not hasattr(sel.target, "index"):
        return sel
    while True:
        artist_xys = sel.artist.get_xydata()
        new_idx = (int(np.ceil(sel.target.index) - 1) if key == "left"
                   else int(np.floor(sel.target.index) + 1) if key == "right"
                   else sel.target.index) % len(artist_xys)
        target = AttrArray(artist_xys[new_idx])
        target.index = new_idx
        sel = sel._replace(target=target, dist=0)
        if np.isfinite(target).all():
            return sel


@move.register(AxesImage)
@_call_with_selection
def _(sel, *, key):
    if type(sel.artist) != AxesImage:
        # All bets are off with subclasses such as NonUniformImage even if they
        # implement `get_cursor_data` because we do not know where a given
        # index maps back physically.
        return sel
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

    The `Selection` is built from ``*args, **kwargs``.

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
    hl.set_offsets(np.where(np.arange(len(offsets))[None] == sel.target.index,
                            offsets, np.nan))
    _set_valid_props(hl, highlight_kwargs)
    return hl
