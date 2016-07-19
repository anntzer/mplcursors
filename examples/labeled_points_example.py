"""Implement mpldatacursor's "point labels" using event handlers.
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

mplcursors.cursor(ax).connect(
    "add",
    lambda sel: sel.annotation.set_text(labels[sel.pick_info.target.index]))

plt.show()

