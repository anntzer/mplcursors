"""
Display a bar's height and name on top of it upon hovering
==========================================================

Using an event handler to change the annotation text and position.
"""

import string
import matplotlib.pyplot as plt
import mplcursors

fig, ax = plt.subplots()
ax.bar(range(9), range(1, 10), align="center")
labels = string.ascii_uppercase[:9]
ax.set(xticks=range(9), xticklabels=labels, title="Hover over a bar")

cursor = mplcursors.cursor(hover=True)
@cursor.connect("add")
def on_add(sel):
    x, y, width, height = sel.artist[sel.target.index].get_bbox().bounds
    sel.annotation.set(
        text="{}: {}".format(x + width / 2, height),
        position=(0, 20))
    sel.annotation.xy = (x + width / 2, y + height)

plt.show()
