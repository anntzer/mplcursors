"""
Hovering and custom formatters.
===============================

Not much to it.
"""

import string
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
np.random.seed(1977)

x, y = np.random.random((2, 26))
labels = string.ascii_lowercase

fig, ax = plt.subplots()
ax.scatter(x, y, s=200)
ax.set_title("Mouse over a point")

mplcursors.cursor(hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

plt.show()
