"""
Illustrates "point label" functionality.
"""
import matplotlib.pyplot as plt
import mplcursors
import numpy as np

labels = ['a', 'b', 'c', 'd', 'e', 'f']
x = np.array([0, 0.05, 1, 2, 3, 4])

# All points on this figure will point labels.
fig, ax = plt.subplots()
line, = ax.plot(x, x, 'ro')
ax.margins(0.1)

def transformer(pick_info):
    pick_info.ann_text = labels[pick_info.target.index]
    return pick_info

mplcursors.cursor(
    ax,
    transformer=lambda pi: pi.replace(ann_text=labels[pi.target.index]))

plt.show()

