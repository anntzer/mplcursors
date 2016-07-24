"""A single annotation is displayed across all axes.
"""

import matplotlib.pyplot as plt
import numpy as np
import mplcursors

plt.figure()

plt.subplot(2,1,1)
plt.title('Note that only one cursor will be displayed')
plt.plot(range(10), 'ro-')

plt.subplot(2,1,2)
dat = np.arange(100).reshape((10,10))
plt.plot(range(10), 'bo')

mplcursors.cursor()

plt.show()
