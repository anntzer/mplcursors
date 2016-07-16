"""
This example demonstrates both draggable annotation boxes and using the
``display="multiple"`` option.
"""
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

data = np.outer(range(10), range(1, 5))

fig, ax = plt.subplots()
ax.set_title('Try dragging the annotation boxes')
ax.plot(data)

mplcursors.cursor(multiple=True).connect(
    "add", lambda sel: sel.annotation.draggable())

plt.show()
