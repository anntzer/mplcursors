"""
mplcursors' core functionality
==============================

... is to add interactive data cursors to a figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import mplcursors

data = np.outer(range(10), range(1, 5))

fig, ax = plt.subplots()
lines = ax.plot(data)
ax.set_title("Click somewhere on a line.\nRight-click to deselect.\n"
             "Annotations can be dragged.")
fig.tight_layout()

mplcursors.cursor(lines)

plt.show()
