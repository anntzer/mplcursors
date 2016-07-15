# FIXME no CS support.
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

fig, ax = plt.subplots()
cf = ax.contour(np.random.random((10,10)))
# For contours, you'll have to explicitly specify the ContourSet ("cf", in this
# case) for the z-values to be displayed. Filled contours aren't properly
# supported, as they only fire a pick even when their edges are selected.
mplcursors.cursor(cf)
plt.show()
