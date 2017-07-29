"""
Changing properties of the popup
================================

Use an event handler to customize the popup.
"""

import matplotlib.pyplot as plt
import numpy as np
import mplcursors

fig, axes = plt.subplots(ncols=2)

left_artist = axes[0].plot(range(11))
axes[0].set(title="No box, different position", aspect=1)

right_artist = axes[1].imshow(np.arange(100).reshape(10, 10))
axes[1].set(title="Fancy white background")

# Make the text pop up "underneath" the line and remove the box...
c1 = mplcursors.cursor(left_artist)
@c1.connect("add")
def _(sel):
    sel.annotation.set(position=(15, -15))
    # Note: Needs to be set separately due to matplotlib/matplotlib#8956.
    sel.annotation.set_bbox(None)

# Make the box have a white background with a fancier connecting arrow
c2 = mplcursors.cursor(right_artist)
@c2.connect("add")
def _(sel):
    sel.annotation.get_bbox_patch().set(fc="white")
    sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=.5)

plt.show()
