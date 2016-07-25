import numpy as np
import matplotlib.pyplot as plt
import mplcursors

x, y, z = np.random.random((3, 10))
fig, axs = plt.subplots(2)
axs[0].scatter(x, y, c=z, s=100*np.random.random(10))
axs[1].plot(x, y, "o")
mplcursors.cursor()
plt.show()
