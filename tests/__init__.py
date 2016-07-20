import os
import matplotlib


# For testing on Travis.
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
