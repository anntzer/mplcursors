"""
Keyboard shortcuts
==================

By default, mplcursors uses "t" to toggle interactivity and "d" to hide/show
annotation boxes.  These shortcuts can be customized.
"""

import matplotlib.pyplot as plt
import mplcursors

fig, ax = plt.subplots()
ax.plot(range(10), "o-")
ax.set_title('Press "e" to enable/disable the datacursor\n'
             'Press "h" to hide/show any annotation boxes')

mplcursors.cursor(bindings={"toggle_visible": "h", "toggle_enabled": "e"})

plt.show()
