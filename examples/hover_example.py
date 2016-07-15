"""
Demonstrates the hover functionality of mpldatacursor as well as point labels
and a custom formatting function. Notice that overlapping points have both
labels displayed.
"""
import string
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
np.random.seed(1977)

x, y = np.random.random((2, 26))

fig, ax = plt.subplots()
ax.scatter(x, y, s=200)
ax.set_title('Mouse over a point')

mplcursors.cursor(hover=True)

plt.show()
