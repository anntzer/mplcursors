from matplotlib import cbook, pyplot as plt
from matplotlib.axes import Axes

from ._mplcursors import Cursor


__all__ = ["cursor"]


def cursor(artists_or_axes=None, **kwargs):
    if artists_or_axes is None:
        artists_or_axes = [
            ax for fig in map(plt.figure, plt.get_fignums()) for ax in fig.axes]
    elif not cbook.iterable(artists_or_axes):
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
