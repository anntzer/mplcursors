from collections import namedtuple
from functools import singledispatch
import warnings

from matplotlib import cbook
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np


class AttrArray(np.ndarray):
    """An array subclass that can store additional attributes.
    """

    def __new__(cls, array):
        return np.asarray(array).view(cls)


Selection = namedtuple("Selection", "artist target dist annotation extras")
Selection.artist.__doc__ = (
    "The selected artist.")
Selection.target.__doc__ = (
    "The point picked within the artist, in data coordinates.")
Selection.dist.__doc__ = (
    "The distance from the click to the target, in pixels.")
Selection.annotation.__doc__ = (
    "The instantiated `matplotlib.text.Annotation`.")
Selection.extras.__doc__ = (
    "An additional list of artists (e.g., highlighters) that will be cleared "
    "at the same time as the annotation.")


@singledispatch
def compute_pick(artist, event):
    """Find whether ``artist`` has been picked by ``event``.

    If it has, return the appropriate `Selection`; otherwise return ``None``.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    warnings.warn("Support for {} is missing".format(type(artist)))


@compute_pick.register(Line2D)
def _(artist, event):
    contains, _ = artist.contains(event)
    if not contains:
        return

    # Always work in screen coordinates, as this is how we need to compute
    # distances.  Note that the artist transform may be different from the axes
    # transform (e.g., for axvline).
    x, y = event.x, event.y
    artist_xs, artist_ys = (
        artist.get_transform().transform(artist.get_xydata()).T)
    drawstyle_conv = {
        "_draw_lines": lambda xs, ys: (xs, ys),
        "_draw_steps_pre": cbook.pts_to_prestep,
        "_draw_steps_mid": cbook.pts_to_midstep,
        "_draw_steps_post": cbook.pts_to_poststep}[
            artist.drawStyles[artist.get_drawstyle()]]
    artist_xs, artist_ys = drawstyle_conv(artist_xs, artist_ys)
    ax = artist.axes
    px_to_data = ax.transData.inverted().transform_point

    # Find the closest vertex.
    d2_vs = (artist_xs - x) ** 2 + (artist_ys - y) ** 2
    vs_argmin = np.argmin(d2_vs)
    vs_min = np.sqrt(d2_vs[vs_argmin])
    vs_target = AttrArray(
        px_to_data((artist_xs[vs_argmin], artist_ys[vs_argmin])))
    vs_target.index = vs_argmin
    vs_info = Selection(artist, vs_target, vs_min, None, None)

    if artist.get_linestyle() in ["None", "none", " ", "", None]:
        return vs_info

    # Find the closest projection.
    # Unit vectors for each segment.
    uxs = artist_xs[1:] - artist_xs[:-1]
    uys = artist_ys[1:] - artist_ys[:-1]
    ds = np.sqrt(uxs ** 2 + uys ** 2)
    uxs /= ds
    uys /= ds
    # Vectors from each vertex to the event.
    dxs = x - artist_xs[:-1]
    dys = y - artist_ys[:-1]
    # Cross-products.
    d_ps = np.abs(dxs * uys - dys * uxs)
    # Dot products.
    dot = dxs * uxs + dys * uys
    # Set the distance to infinity if the projection is not in the segment.
    d_ps[~((0 < dot) & (dot < ds))] = np.inf
    ps_argmin = np.argmin(d_ps)
    ps_min = d_ps[ps_argmin]

    if vs_min < ps_min:
        return vs_info
    else:
        p_x = artist_xs[ps_argmin] + dot[ps_argmin] * uxs[ps_argmin]
        p_y = artist_ys[ps_argmin] + dot[ps_argmin] * uys[ps_argmin]
        ps_target = AttrArray(px_to_data((p_x, p_y)))
        if artist.drawStyles[artist.get_drawstyle()] == "_draw_lines":
            ps_target.index = ps_argmin + dot[ps_argmin] / ds[ps_argmin]
        ps_info = Selection(artist, ps_target, ps_min, None, None)
        return ps_info


@compute_pick.register(PathCollection)
def _(artist, event):
    contains, info = artist.contains(event)
    if not contains:
        return
    # Snapping, really only works for scatter plots.
    ax = artist.axes
    idxs = info["ind"]
    offsets = artist.get_offsets()[idxs]
    d2 = ((ax.transData.transform(offsets) -
           [event.x, event.y]) ** 2).sum(axis=1)
    argmin = d2.argmin()
    target = AttrArray(offsets[argmin])
    target.index = idxs[argmin]
    return Selection(artist, target, np.sqrt(d2[argmin]), None, None)


@compute_pick.register(AxesImage)
@compute_pick.register(Patch)
def _(artist, event):
    contains, _ = artist.contains(event)
    if not contains:
        return
    return Selection(artist, (event.xdata, event.ydata), 0, None, None)


@compute_pick.register(Text)
def _(artist, event):
    return


@singledispatch
def get_ann_text(*args):
    """Compute an annotating text for a `Selection` (unpacked as ``*args``).

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    sel = Selection(*args)
    warnings.warn("Support for {} is missing".format(type(sel.artist)))
    return ""


@get_ann_text.register(Line2D)
@get_ann_text.register(PathCollection)
@get_ann_text.register(Patch)
def _(*args):
    sel = Selection(*args)
    ax = sel.artist.axes
    x, y = sel.target
    label = sel.artist.get_label()
    if label.startswith("_"):
        return "x: {}\ny: {}".format(
            ax.format_xdata(x), ax.format_ydata(y))
    else:
        return "{}\nx: {}\ny: {}".format(
            label, ax.format_xdata(x), ax.format_ydata(y))


@get_ann_text.register(AxesImage)
def _(*args):
    sel = Selection(*args)
    artist = sel.artist
    ax = artist.axes
    x, y = sel.target
    event = namedtuple("event", "xdata ydata")(x, y)
    return "x: {}\ny: {}\nz: {}".format(ax.format_xdata(x),
                                        ax.format_ydata(y),
                                        artist.get_cursor_data(event))


@singledispatch
def move(*args, by):
    """"Move" a `Selection` by an appropriate "distance".

    This function is used to implement annotation displacement through the
    keyboard.

    This is a single-dispatch function; implementations for various artist
    classes follow.
    """
    return Selection(*args)


@move.register(Line2D)
def _(*args, by):
    sel = Selection(*args)
    if not hasattr(sel.target, "index"):
        return sel
    if by < 0:
        new_idx = int(np.ceil(sel.target.index) + by)
    elif by > 0:
        new_idx = int(np.floor(sel.target.index) + by)
    artist_xys = sel.artist.get_xydata()
    target = AttrArray(artist_xys[new_idx % len(artist_xys)])
    target.index = new_idx
    return sel._replace(target=target, dist=0)
