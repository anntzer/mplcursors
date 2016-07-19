"""Display a bar's height and name on top of it upon hovering.

An example of using PickInfo transformers to change `ann_text` and `target`.
"""

import string
import matplotlib.pyplot as plt
import mplcursors

fig, ax = plt.subplots()
ax.bar(range(9), range(1, 10), align='center')
labels = string.ascii_uppercase[:9]
ax.set(xticks=range(9), xticklabels=labels, title='Hover over a bar')

def transform(pick_info):
    x, y, width, height = pick_info.artist.get_bbox().bounds
    pi = pick_info.replace(target=(x + width / 2, y + height),
                           ann_text="{}: {}".format(x + width / 2, height))
    return pi

mplcursors.cursor(
    hover=True, annotation_kwargs=dict(xytext=(0, 20)), transformer=transform)

plt.show()
