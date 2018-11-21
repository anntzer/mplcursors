next
====

- Display value in annotation for colormapped scatter plots.
- The add/remove callbacks no longer rely on Matplotlib's `CallbackRegistry`.
  `Cursor.connect` now returns the callback itself (simplifying its use as a
  decorator).  `Cursor.disconnect` now takes two arguments: the event name and
  the callback function.  Strong references are kept for the callbacks.

0.2.1
=====

No new features; test suite updated for compatibility with Matplotlib 3.0.

Miscellaneous bugfixes.

0.2
===

- Updated dependency to Matplotlib 2.1 (2.0 gives more information about
  orientation of bar plots; 2.1 improves the handling of step plots).
- Setting `MPLCURSORS` hooks `Figure.draw` (once per figure only) instead of
  `plt.show`, thus supporting figures created after the first call to
  `plt.show`.
- Automatic positioning and alignment of annotation text.
- Selections on images now have an index as well.
- Selections created on `scatter` plots, `errorbar` plots, and `polar` plots
  can now be moved.
- `PathCollection`\s not created by `plt.scatter` are now picked as paths, not
  as collections of points.
- `Patch`\es now pick on their borders, not their interior.
- Improved picking of `Container`\s.
- In hover mode, annotations can still be removed by right-clicking.

Miscellaneous bugfixes.

0.1
===

- First public release.
