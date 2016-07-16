from collections import namedtuple
from functools import singledispatch

from matplotlib import cbook
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


class PickInfo(namedtuple("_PickInfo", "artist dist target")):
    @property
    def ann_text(self):
        try:
            return self._ann_text
        except AttributeError:
            self._ann_text = get_ann_text(*self)
            return self._ann_text

    @ann_text.setter
    def ann_text(self, value):
        self._ann_text = value


@singledispatch
def compute_pick(artist, event):
    raise NotImplementedError("Support for {} is missing".format(type(artist)))


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
    d2_verts = (artist_xs - x) ** 2 + (artist_ys - y) ** 2
    verts_argmin = np.argmin(d2_verts)
    verts_min = np.sqrt(d2_verts[verts_argmin])
    verts_target = px_to_data((artist_xs[verts_argmin],
                                 artist_ys[verts_argmin]))
    verts_info = PickInfo(artist, verts_min, verts_target)

    if artist.get_linestyle() in ["None", "none", " ", "", None]:
        return verts_info

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
    d_projs = np.abs(dxs * uys - dys * uxs)
    # Dot products.
    dot = dxs * uxs + dys * uys
    # Set the distance to infinity if the projection is not in the segment.
    d_projs[~((0 < dot) & (dot < np.sqrt(dxs ** 2 + dys ** 2)))] = np.inf
    projs_argmin = np.argmin(d_projs)
    projs_min = d_projs[projs_argmin]

    if verts_min < projs_min:
        return verts_info
    else:
        proj_x = artist_xs[projs_argmin] + dot[projs_argmin] * uxs[projs_argmin]
        proj_y = artist_ys[projs_argmin] + dot[projs_argmin] * uys[projs_argmin]
        projs_target = px_to_data((proj_x, proj_y))
        projs_info = PickInfo(artist, projs_min, projs_target)
        return projs_info


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
    return PickInfo(artist, np.sqrt(d2[argmin]), offsets[argmin])


@compute_pick.register(AxesImage)
@compute_pick.register(Patch)
def _(artist, event):
    contains, _ = artist.contains(event)
    if not contains:
        return
    return PickInfo(artist, 0, (event.xdata, event.ydata))


@singledispatch
def get_ann_text(artist, dist, target):
    raise NotImplementedError("Support for {} is missing".format(type(artist)))


@get_ann_text.register(Line2D)
@get_ann_text.register(PathCollection)
@get_ann_text.register(Patch)
def _(artist, dist, target):
    ax = artist.axes
    x, y = target
    label = artist.get_label()
    if label.startswith("_"):
        return "x: {}\ny: {}".format(ax.format_xdata(x), ax.format_ydata(y))
    else:
        return "{}\nx: {}\ny: {}".format(
            label, ax.format_xdata(x), ax.format_ydata(y))


@get_ann_text.register(AxesImage)
def _(artist, dist, target):
    artist = artist
    ax = artist.axes
    x, y = target
    event = namedtuple("event", "xdata ydata")(x, y)
    return "x: {}\ny: {}\nz: {}".format(ax.format_xdata(x),
                                        ax.format_ydata(y),
                                        artist.get_cursor_data(event))
