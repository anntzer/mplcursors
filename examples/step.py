"""
Step plots
==========

A selection on a step plot holds precise information on the x and y position
in the ``sel.target.index`` sub-attribute.
"""

from matplotlib import pyplot as plt
import mplcursors
import numpy as np


fig, axs = plt.subplots(4, sharex=True, sharey=True)
np.random.seed(42)
xs = np.arange(5)
ys = np.random.rand(5)

axs[0].plot(xs, ys, "-o")
axs[1].plot(xs, ys, "-o", drawstyle="steps-pre")
axs[2].plot(xs, ys, "-o", drawstyle="steps-mid")
axs[3].plot(xs, ys, "-o", drawstyle="steps-post")
for ax in axs:
    ax.label_outer()

mplcursors.cursor().connect(
    "add",
    lambda sel: sel.annotation.set_text(format(sel.target.index, ".2f")))
plt.show()
