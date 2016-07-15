from matplotlib import pyplot as plt
from mplcursors import Cursor

fig, ax = plt.subplots()
l,  = ax.plot([1, 2, 3])
Cursor([l], highlight=True)
plt.show()
