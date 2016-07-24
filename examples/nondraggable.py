"""Using multiple annotations and disabling draggability via signals.
"""

import matplotlib.pyplot as plt
import numpy as np
import mplcursors

data = np.outer(range(10), range(1, 5))

fig, ax = plt.subplots()
ax.set_title("The annotation boxes are not draggable here.")
ax.plot(data)

mplcursors.cursor(multiple=True).connect(
    "add", lambda sel: sel.annotation.draggable(False))

plt.show()
