"""
Cursors on images
=================

... display the underlying data value.
"""

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

data = np.arange(100).reshape((10, 10))

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(data, origin="lower")
axs[1].imshow(data, origin="upper", extent=[2, 3, 4, 5])

axs[1].set(xlim=(2, 4), ylim=(4, 6))
axs[1].add_artist(AnnotationBbox(OffsetImage(data), (3.5, 5.5)))

mplcursors.cursor()

fig.suptitle("Click anywhere on the image")

plt.show()
