from sphinx_gallery.sorting import ExampleTitleSortKey

class CustomSortKey(ExampleTitleSortKey):
    def __call__(self, filename):
        return ("" if filename == "basic.py"  # goes first
                else super().__call__(filename))
