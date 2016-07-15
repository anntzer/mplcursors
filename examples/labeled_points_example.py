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
mplcursors.cursor(
    ax,
    format=lambda info: "{}\n{}".format(
        info, labels[np.argmin(np.abs(x - info.target[0]))]))

plt.show()

