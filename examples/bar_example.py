"""
A bar plot where each bar's height and name will be displayed above the top of
the bar when it is moused over.  This serves as an example of overriding the
x,y position of the "popup" annotation using the `props_override` option.
"""
import string
import matplotlib.pyplot as plt
import mplcursors

fig, ax = plt.subplots()
ax.bar(range(9), range(1, 10), align='center')
labels = string.ascii_uppercase[:9]
ax.set(xticks=range(9), xticklabels=labels, title='Hover over a bar')

def transform(pick_info):
    x, y, width, height = pick_info.artist.get_bbox().bounds
    pi = pick_info._replace(target=(x + width / 2, y + height))
    pi.ann_text = "{}: {}".format(x + width / 2, height)
    return pi

mplcursors.cursor(
    hover=True, annotation_kwargs=dict(xytext=(0, 20)), transformer=transform)

plt.show()
