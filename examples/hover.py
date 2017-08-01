"""
Annotate on hover
=================

When ``hover`` is set to ``True``, annotations are displayed when the mouse
hovers over the artist, without the need for clicking.
"""

import matplotlib.pyplot as plt
import numpy as np
import mplcursors
np.random.seed(42)

fig, ax = plt.subplots()
ax.scatter(*np.random.random((2, 26)))
ax.set_title("Mouse over a point")

mplcursors.cursor(hover=True)

plt.show()
