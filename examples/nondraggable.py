"""
Using multiple annotations and disabling draggability via signals
=================================================================

By default, each `Cursor` will ever display one annotation at a time.  Pass
``multiple=True`` to display multiple annotations.

Annotations can be made non-draggable by hooking their creation.
"""

import matplotlib.pyplot as plt
import numpy as np
import mplcursors

data = np.outer(range(10), range(1, 5))

fig, ax = plt.subplots()
ax.set_title("Multiple non-draggable annotations")
ax.plot(data)

mplcursors.cursor(multiple=True).connect(
    "add", lambda sel: sel.annotation.draggable(False))

plt.show()
