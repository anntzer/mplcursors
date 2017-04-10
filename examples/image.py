"""
Cursors on images
=================

... display the underlying data value.
"""

import matplotlib.pyplot as plt
import numpy as np
import mplcursors

data = np.arange(100).reshape((10, 10))

fig, axes = plt.subplots(ncols=2)
axes[0].imshow(data, interpolation="nearest", origin="lower")
axes[1].imshow(data, interpolation="nearest", origin="upper",
                     extent=[200, 300, 400, 500])
mplcursors.cursor()

fig.suptitle("Click anywhere on the image")

plt.show()
