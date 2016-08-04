"""A single annotation is displayed across all axes.
"""

import matplotlib.pyplot as plt
import numpy as np
import mplcursors

fig, axs = plt.subplots(2)
fig.suptitle('Note that only one cursor will be displayed')
axs[0].plot(range(10), 'ro-')
axs[1].plot(range(10), 'bo')
mplcursors.cursor()
plt.show()
