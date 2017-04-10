"""
Labeling points from a `DataFrame` scatter plot using a third column
====================================================================

`DataFrame`\\s can be used similarly to any other kind of input.
"""

from matplotlib import pyplot as plt
import mplcursors
from pandas import DataFrame


df = DataFrame(
    [("Alice", 163, 54),
     ("Bob", 174, 67),
     ("Charlie", 177, 73),
     ("Diane", 168, 57)],
    columns=["name", "height", "weight"])

df.plot.scatter("height", "weight")
mplcursors.cursor().connect(
    "add", lambda sel: sel.annotation.set_text(df["name"][sel.target.index]))
plt.show()

# test: skip
