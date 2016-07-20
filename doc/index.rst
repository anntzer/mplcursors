.. mplcursors documentation master file, created by
   sphinx-quickstart on Tue Jul 19 20:06:56 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mplcursors' documentation!
=====================================

:mod:`mplcursors` provide interactive, clickable annotation for
:mod:`matplotlib`.  It is heavily inspired from :mod:`mpldatacursor`
(https://github.com/joferkington/mpldatacursor), with a much simplified API.

Note that :mod:`mplcursors` require Python>=3.5 and :mod:`matplotlib`\>=1.5.0.
There are no plans to support earlier versions; in fact, the minimum version
of :mod:`matplotlib` will likely be raised to 2.1 once it is released (as it
will fix some `bugs`_ related to event handling).

.. _bugs: https://github.com/matplotlib/matplotlib/pull/6497

Basic example
-------------

Basic examples work similarly to :mod:`mpldatacursor`::

    import matplotlib.pyplot as plt
    import numpy as np
    import mplcursors

    data = np.outer(range(10), range(1, 5))

    fig, ax = plt.subplots()
    lines = ax.plot(data)
    ax.set_title("Click somewhere on a line\nRight-click to deselect\n"
                 "Annotations can be dragged.")

    mplcursors.cursor(lines)

    plt.show()

.. image:: /images/basic.png

The `cursor` convenience function makes a collection of artists selectable.
Specifically, its first argument can either be a list of artists or axes (in
which case all artists in each of the axes become selectable); or one can just
pass no argument, in which case all artists in all figures become selectable.
Other :class:`arguments <mplcursors.Cursor>` (which are all keyword-only)
allow for basic customization of the `Cursor`'s behavior; please refer to the
constructor's documentation.

Customization
-------------

Instead of providing a host of keyword arguments in `Cursor`'s constructor,
:mod:`mplcursors` represents selections as `Selection` namedtuples and lets you
hook into their addition and removal.

Specifically, a `Selection` has the following fields:

    - :attr:`artist`: the selected artist,
    - :attr:`target`: the point picked within the artist; if a point is picked
      on a `matplotlib Line2D <matplotlib.lines.Line2D>`, the index of the
      point is available as the :attr:`target.index` sub-attribute.
    - :attr:`dist`: the distance from the point clicked to the :attr:`target`
      (mostly used to decide which ).
    - :attr:`annotation`: a `matplotlib Annotation
      <matplotlib.text.Annotation>` object.
    - :attr:`extras`: an additional list of artists, that will be removed
      whenever the main :attr:`annotation` is deselected.

Thus, in order to customize, e.g., the annotation text, one can call::

    lines = ax.plot(range(3), range(3), "o")
    labels = ["a", "b", "c"]
    cursor = mplcursors.cursor(lines)
    cursor.connect(
        "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

Whenever a point is selected (resp. deselected), the ``"add"`` (resp.
``"remove"``) event is triggered and the registered callbacks are executed,
with the `Selection` as only argument.  Here, the only callback updates the
text of the annotation to a per-point label. (``cursor.connect("add")`` can
also be used as a decorator to register a callback, see below for an example.)

For additional customizations of the position and appearance of the annotation,
see :file:`bar_example.py` and :file:`change_popup_color.py` in the
:file:`examples` folder of the source tree.

Callbacks can also be used to make additional changes to the figure when
a selection occurs.  For example, the following snippet (extracted from
:file:`multi_highlight_example.py`) ensures that whenever an artist is
selected, another artist that has been "paired" with it (via the ``pairs`` map)
also gets selected::

    @cursor.connect("add")
    def on_add(sel):
        sel.extras.append(cursor.add_highlight(pairs[sel.artist]))

Note that the paired artist will also get de-highlighted when the "first"
artist is deselected.


.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

