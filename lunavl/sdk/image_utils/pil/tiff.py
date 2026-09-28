"""
Module contains pillow-simd workaround for the issue https://github.com/python-pillow/Pillow/issues/3690 (pickle tiff).
"""

from PIL.Image import register_open
from PIL.ImageFile import ImageFile
from PIL.TiffImagePlugin import TiffImageFile, _accept


class TiffImageFileUseLibtiff(TiffImageFile):  # pylint: disable-msg=W0223
    """
    Overload TiffImageFile class to workaround the issue https://github.com/python-pillow/Pillow/issues/3690.

    Bug description:
       Object TiffImageFile has not attributes _n_frames and _is_animated after using asyncio function run_in_executor
       (this function use pickle for execution in sub processes). And method numpy.array(image) is broken.

    Fix:
       Overload ``TiffImageFile.load()`` method and register the custom tiff image file plugin.
    """

    def load(self):
        # ``self.tile`` is empty once the image is decoded. Without this check a nested ``load()`` call
        # (recent Pillow calls it from ``load_end`` via ``ImageOps.exif_transpose``) re-enters
        # ``_load_libtiff`` and fails with "Not exactly one tile".
        if self.tile and getattr(self, "use_load_libtiff", False):
            return self._load_libtiff()
        return ImageFile.load(self)


def applyTiffPluginFix():
    # Register the custom tiff image file plugin
    register_open(TiffImageFile.format, TiffImageFileUseLibtiff, _accept)
