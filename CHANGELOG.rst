Changelog
=========

0.5.3
-----

- Require Python 3.7 (due to setuptools support ranges); mark Matplotlib 3.7.1
  as incompatible.
- Highlights can be removed by right-clicking anywhere on the highlighting
  artist, not only on the annotation.

0.5.2
-----

- Fix compatibility with Matplotlib 3.6 and with PEP517 builds.
- Non-multiple cursors can now be dragged.

0.5.1
-----

No new features; minor changes to docs.

0.5
---

- **Breaking change**: ``index`` is now a direct attribute of the `Selection`,
  rather than a sub-attribute via ``target``.  (``Selection.target.index`` has
  been deprecated and will be removed in the future.)
- Additional annotations are no longer created when dragging a ``multiple``
  cursor.
- Clicking on an annotation also updates the "current" selection for keyboard
  motion purposes.
- Disabling a cursor also makes it unresponsive to motion keys.
- Hovering is still active when the pan or zoom buttons are pressed (but not if
  there's a pan or zoom currently being selected).
- Annotations are now :class:`~matplotlib.figure.Figure`-level artists, rather
  than Axes-level ones (so as to be drawn on top of twinned axes, if present).

0.4
---

- Invisible artists are now unpickable (patch suggested by @eBardieCT).
- The ``bindings`` kwarg can require modifier keys for mouse button events.
- Transient hovering (suggested by @LaurenceMolloy).
- Switch to supporting only "new-style"
  (:class:`~matplotlib.collections.LineCollection`)
  :meth:`~matplotlib.axes.Axes.stem` plots.
- Cursors are drawn with ``zorder=np.inf``.

0.3
---

- Updated dependency to Matplotlib 3.1 (``Annotation.{get,set}_anncoords``),
  and thus Python 3.6, numpy 1.11.
- Display value in annotation for colormapped scatter plots.
- Improve formatting of image values.
- The add/remove callbacks no longer rely on Matplotlib's
  :class:`~matplotlib.cbook.CallbackRegistry`.  `Cursor.connect` now returns
  the callback itself (simplifying its use as a decorator).
  `Cursor.disconnect` now takes two arguments: the event name and the callback
  function.  Strong references are kept for the callbacks.
- Overlapping annotations are now removed one at a time.
- Re-clicking on an already selected point does not create a new annotation
  (patch suggested by @schneeammer).
- :class:`~matplotlib.collections.PatchCollection`\s are now pickable (on their
  borders) (patch modified from a PR by @secretyv).
- Support :class:`~matplotlib.collections.Collection`\s where
  :meth:`~matplotlib.collections.Collection.get_offset_transform()` is not
  ``transData`` (patch suggested by @yt87).
- Support setting both ``hover`` and ``multiple``.
- The ``artist`` attribute of Selections is correctly set to the
  :class:`~matplotlib.container.Container` when picking a
  :class:`~matplotlib.container.Container`, rather than to the internally used
  wrapper.

0.2.1
-----

No new features; test suite updated for compatibility with Matplotlib 3.0.

Miscellaneous bugfixes.

0.2
---

- Updated dependency to Matplotlib 2.1 (2.0 gives more information about
  orientation of bar plots; 2.1 improves the handling of step plots).
- Setting :envvar:`MPLCURSORS` hooks `Figure.draw
  <matplotlib.figure.Figure.draw>` (once per figure only) instead of `plt.show
  <matplotlib.pyplot.show>`, thus supporting figures created after the first
  call to `plt.show <matplotlib.pyplot.show>`.
- Automatic positioning and alignment of annotation text.
- Selections on images now have an index as well.
- Selections created on :meth:`~matplotlib.axes.Axes.scatter` plots,
  :meth:`~matplotlib.axes.Axes.errorbar` plots, and
  :meth:`~matplotlib.axes.Axes.polar` plots can now be moved.
- :class:`~matplotlib.collections.PathCollection`\s not created by
  :meth:`~matplotlib.axes.Axes.scatter` are now picked as paths, not as
  collections of points.
- :class:`~matplotlib.patches.Patch`\es now pick on their borders, not their
  interior.
- Improved picking of :class:`~matplotlib.container.Container`\s.
- In hover mode, annotations can still be removed by right-clicking.

Miscellaneous bugfixes.

0.1
---

- First public release.
