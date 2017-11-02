"""
Linked artists
==============

An example of connecting to cursor events: when an artist is selected, also
highlight its "partner".
"""

import numpy as np
import matplotlib.pyplot as plt
import mplcursors


def main():
    fig, axes = plt.subplots(ncols=2)
    num = 5
    xy = np.random.random((num, 2))

    lines = []
    for i in range(num):
        line, = axes[0].plot((i + 1) * np.arange(10))
        lines.append(line)

    points = []
    for x, y in xy:
        point, = axes[1].plot([x], [y], linestyle="none", marker="o")
        points.append(point)

    cursor = mplcursors.cursor(points + lines, highlight=True)
    pairs = dict(zip(points, lines))
    pairs.update(zip(lines, points))

    @cursor.connect("add")
    def on_add(sel):
        sel.extras.append(cursor.add_highlight(pairs[sel.artist]))

    plt.show()


if __name__ == "__main__":
    main()
