Welcome to mplcursors' documentation!
=====================================

:mod:`mplcursors` provides interactive data selection cursors for Matplotlib_.
It is inspired from mpldatacursor_, with a much simplified API.

.. _Matplotlib: http://matplotlib.org
.. _mpldatacursor: https://github.com/joferkington/mpldatacursor

:mod:`mplcursors` requires Python 3, and Matplotlib≥2.1.

.. _installation:

Installation
------------

Pick one among:

.. code-block:: sh

    $ pip install mplcursors  # from PyPI
    $ pip install git+https://github.com/anntzer/mplcursors  # from Github

.. _basic-example:

Basic example
-------------

Basic examples work similarly to mpldatacursor_::

   import matplotlib.pyplot as plt
   import numpy as np
   import mplcursors

   data = np.outer(range(10), range(1, 5))

   fig, ax = plt.subplots()
   lines = ax.plot(data)
   ax.set_title("Click somewhere on a line.\nRight-click to deselect.\n"
                "Annotations can be dragged.")

   mplcursors.cursor(lines) # or just mplcursors.cursor()

   plt.show()

.. image:: /images/basic.png

The `cursor` convenience function makes a collection of artists selectable.
Specifically, its first argument can either be a list of artists or axes (in
which case all artists in each of the axes become selectable); or one can just
pass no argument, in which case all artists in all figures become selectable.
Other arguments (which are all keyword-only) allow for basic customization of
the `Cursor`’s behavior; please refer to that class' documentation.

.. _activation-by-environment-variable:

Activation by environment variable
----------------------------------

It is possible to use :mod:`mplcursors` without modifying *any* source code:
setting the :envvar:`MPLCURSORS` environment variable to a JSON-encoded dict
will patch `Figure.draw <matplotlib.figure.Figure.draw>` to automatically
call `cursor` (with the passed keyword arguments, if any) after the figure is
drawn for the first time (more precisely, after the first draw that includes a
selectable artist). Typical settings include::

   $ MPLCURSORS={} python foo.py

and::

   $ MPLCURSORS='{"hover": 1}' python foo.py

Note that this will only work if :mod:`mplcursors` has been installed, not if
it is simply added to the :envvar:`PYTHONPATH`.

Note that this will not pick up artists added to the figure after the first
draw, e.g. through interactive callbacks.

.. _default-ui:

Default UI
----------

- A left click on a line (a point, for plots where the data points are not
  connected) creates a draggable annotation there.  Only one annotation is
  displayed (per `Cursor` instance), except if the ``multiple`` keyword
  argument was set.
- A right click on an existing annotation will remove it.
- Clicks do not trigger annotations if the zoom or pan tool are active.  It is
  possible to bypass this by *double*-clicking instead.
- For annotations pointing to lines or images, :kbd:`Shift-Left` and
  :kbd:`Shift-Right` move the cursor "left" or "right" by one data point.  For
  annotations pointing to images, :kbd:`Shift-Up` and :kbd:`Shift-Down` are
  likewise available.
- :kbd:`v` toggles the visibility of the existing annotation(s).
- :kbd:`e` toggles whether the `Cursor` is active at all (if not, no event
  other than re-activation is propagated).

These bindings are all customizable via `Cursor`’s ``bindings`` keyword
argument.  Note that the keyboard bindings are only active if the canvas has
the keyboard input focus.

.. _customization:

Customization
-------------

Instead of providing a host of keyword arguments in `Cursor`’s constructor,
:mod:`mplcursors` represents selections as `Selection` objects (essentially,
namedtuples) and lets you hook into their addition and removal.

Specifically, a `Selection` has the following fields:

- :attr:`~.artist`: the selected artist,
- :attr:`~.target`: the point picked within the artist; if a point is
  picked on a :class:`~matplotlib.lines.Line2D`, the index of the point is
  available as the :attr:`target.index` sub-attribute (for more details, see
  `selection-indices`).
- :attr:`~.dist`: the distance from the point clicked to the :attr:`~.target`
  ly used to decide which artist to select).
- :attr:`~.annotation`: a Matplotlib :class:`~matplotlib.text.Annotation`
  object.
- :attr:`~.extras`: an additional list of artists, that will be removed
  whenever the main :attr:`~.annotation` is deselected.

For certain classes of artists, additional information about the picked point
is available in the :attr:`target.index` sub-attribute:

- For :class:`~matplotlib.lines.Line2D`\s, it contains the index of the
  selected point (see `selection-indices` for more details, especially
  regarding step plots).
- For :class:`~matplotlib.image.AxesImage`\s, it contains the ``(y, x)``
  indices of the selected point (such that ``data[y, x]`` in that order is the
  value at that point; note that this means that the indices are not in the
  same order as the target coordinates!).
- For :class:`~matplotlib.container.Container`\s, it contains the index of the
  selected sub-artist.
- For :class:`~matplotlib.collections.LineCollection`\s and
  :class:`~matplotlib.collections.PathCollection`\s, it contains a pair: the
  index of the selected line, and the index within the line, as defined above.

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
For an example using pandas' `DataFrame <pandas.DataFrame>`\s, see
`/examples/dataframe`.

For additional examples of customization of the position and appearance of the
annotation, see `/examples/bar` and `/examples/change_popup_color`.

.. note::
   When the callback is fired, the position of the annotating text is
   temporarily set to ``(nan, nan)``.  This allows us to track whether a
   callback explicitly sets this position, and, if none does, automatically
   compute a suitable position.

   Likewise, if the text alignment is not explicitly set but the position is,
   then a suitable alignment will be automatically computed.

Callbacks can also be used to make additional changes to the figure when
a selection occurs.  For example, the following snippet (extracted from
`/examples/paired_highlight`) ensures that whenever an artist is selected,
another artist that has been "paired" with it (via the ``pairs`` map) also gets
selected::

   @cursor.connect("add")
   def on_add(sel):
       sel.extras.append(cursor.add_highlight(pairs[sel.artist]))

Note that the paired artist will also get de-highlighted when the "first"
artist is deselected.

In order to set the status bar text from a callback, it may be helpful to
clear it during "normal" mouse motion, e.g.::

   fig.canvas.mpl_connect(
       "motion_notify_event",
       lambda event: fig.canvas.toolbar.set_message(""))
   cursor = mplcursors.cursor(hover=True)
   cursor.connect(
       "add",
       lambda sel: fig.canvas.toolbar.set_message(
           sel.annotation.get_text().replace("\n", "; ")))

.. _selection-indices:

Selection indices
-----------------

When picking a point on a "normal" line, the target index has an integer part
equal to the index of segment it is on, and a fractional part that indicates
where the point is within that segment.

For step plots (i.e., created by `plt.step <matplotlib.pyplot.step>` or
`plt.plot <matplotlib.pyplot.plot>`\ ``(..., drawstyle="steps-...")``, we
return a special :class:`Index` object, with attributes :attr:`int` (the
segment index), :attr:`x` (how far the point has advanced in the ``x``
direction) and :attr:`y` (how far the point has advanced in the ``y``
direction).  See `/examples/step` for an example.

On polar plots, lines can be either drawn with a "straight" connection between
two points (in screen space), or "curved" (i.e., using linear interpolation in
data space).  In the first case, the fractional part of the index is defined as
for cartesian plots.  In the second case, the index in computed first on the
interpolated path, then divided by the interpolation factor (i.e., pretending
that each interpolated segment advances the same index by the same amount).

.. _complex-plots:

Complex plots
-------------

Some complex plots, such as contour plots, may be partially supported,
or not at all.  Typically, it is because they do not subclass
:class:`~matplotlib.artist.Artist`, and thus appear to `cursor` as a collection
of independent artists (each contour level, in the case of contour plots).

It is usually possible, again, to hook the ``"add"`` signal to provide
additional information in the annotation text.  See `/examples/contour` for an
example.

Animations
----------

Matplotlib's :mod:`~.animation` blitting mode assumes that the animation
object is entirely in charge of deciding what artists to draw and when.  In
particular, this means that the ``animated`` property is set on certain
artists.  As a result, when :mod:`mplcursors` tries to blit an animation on top
of the image, the animated artists will not be drawn, and disappear.  More
importantly, it also means that once an annotation is added, :mod:`mplcursors`
cannot remove it (as it needs to know what artists to redraw to restore the
original state).

As a workaround, either switch off blitting, or unset the ``animated`` property
on the relevant artists before using a cursor.  (The only other fix I can
envision is to walk the entire tree of artists, record their visibility status,
and try to later restore them; but this would fail for `~.ArtistAnimation`\s
which themselves fiddle with artist visibility).

Indices and tables
==================

* `genindex`
* `modindex`
* `search`

.. toctree::
   :hidden:

   Main page <self>
   API <mplcursors>
   Examples <examples/index>
