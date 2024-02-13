"""Module for creating warped images
"""

from typing import Optional, Union

from FaceEngine import IHumanWarperPtr  # pylint: disable=E0611,E0401
from FaceEngine import Image as CoreImage  # pylint: disable=E0611,E0401
from numpy import ndarray
from PIL.Image import Image as PilImage

from lunavl.sdk.detectors.bodydetector import BodyDetection
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.image_utils.image import ColorFormat, VLImage


class BodyWarpedImage(VLImage):
    """
    Raw warped image.
    Properties of a warped image:

        - it's always in RGB color format
        - it always contains just a single human body
    """

    def __init__(
        self,
        body: Union[bytes, bytearray, ndarray, PilImage, CoreImage, VLImage],
        filename: str = "",
        colorFormat: Optional[ColorFormat] = None,
    ):
        """
        Init.

        Args:
            body: body of image - bytes/bytearray or core image or pil image or vlImage
            filename: user mark a source of image
            colorFormat: output image color format
        """
        if isinstance(body, VLImage):
            self.source = body.source
            self.filename = body.filename
            self.coreImage = body.coreImage
        else:
            super().__init__(body, filename=filename, colorFormat=colorFormat)
        self.assertWarp()

    def assertWarp(self):
        """
        Validate size and format

        Raises:
            ValueError("Bad image format for warped image, must be R8G8B8"): if image has incorrect format
        Warnings:
            this checks are not guarantee that image is warp. This function is intended for debug
        """
        if self.rect.size.height != 256 or self.rect.width != 128:
            raise ValueError("Bad image size for body warped image")
        if self.format != self.format.R8G8B8:
            raise ValueError("Bad image format for warped image, must be R8G8B8")

    #  pylint: disable=W0221
    @classmethod
    def load(cls, *, filename: Optional[str] = None, url: Optional[str] = None) -> "BodyWarpedImage":  # type: ignore
        """
        Load image from numpy array or file or url.

        Args:
            filename: filename
            url: url

        Returns:
            warp
        """
        warp = cls(body=VLImage.load(filename=filename, url=url), filename=filename or "")
        warp.assertWarp()
        return warp

    @property
    def warpedImage(self) -> "BodyWarpedImage":
        """
        Property for compatibility with *Warp* for outside methods.
        Returns:
            self
        """
        return self


class BodyWarp:
    """
    Structure for storing warp.

    Attributes:
        sourceDetection (BodyDetection): detection which generated warp
        warpedImage (BodyWarpedImage):  warped image
    """

    __slots__ = ["sourceDetection", "warpedImage"]

    def __init__(self, warpedImage: BodyWarpedImage, sourceDetection: BodyDetection):
        """
        Init.

        Args:
            warpedImage: warped image
            sourceDetection: detection which generated warp
        """
        self.sourceDetection = sourceDetection
        self.warpedImage = warpedImage


class BodyWarper:
    """
    Class warper.

    Attributes:
        _coreWarper (IWarperPtr): core warper
    """

    __slots__ = ["_coreWarper"]

    def __init__(self, coreWarper: IHumanWarperPtr):
        """
        Init.

        Args:
            coreWarper: core warper
        """
        self._coreWarper = coreWarper

    def warp(self, humanDetection: BodyDetection) -> BodyWarp:
        """
        Create warp from detection.

        Args:
            humanDetection: human body detection with landmarks17

        Returns:
            Warp
        Raises:
            LunaSDKException: if creation failed
        """
        error, warp = self._coreWarper.warp(humanDetection.image.coreImage, humanDetection.coreEstimation.detection)
        assertError(error)

        warpedImage = BodyWarpedImage(body=warp, filename=humanDetection.image.filename)

        return BodyWarp(warpedImage, humanDetection)
