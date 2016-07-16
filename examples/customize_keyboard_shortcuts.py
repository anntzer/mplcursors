"""
An example of how to customize the keyboard shortcuts.
By default mpldatacursor will use "t" to toggle interactivity and "d" to
hide/delete annotation boxes.
"""
import matplotlib.pyplot as plt
import mplcursors

fig, ax = plt.subplots()
ax.plot(range(10), 'bo-')
ax.set_title('Press "e" to enable/disable the datacursor\n'
             'Press "h" to hide/show any annotation boxes')

mplcursors.cursor(bindings={"toggle_visibility": "h", "toggle_enabled": "e"})

plt.show()
