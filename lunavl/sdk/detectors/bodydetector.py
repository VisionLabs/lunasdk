"""
Module contains function for detection human bodies on images.
"""

from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, overload

from FaceEngine import Detection, FSDKErrorResult, Human, HumanDetectionType  # pylint: disable=E0611,E0401

from ..async_task import AsyncTask
from ..errors.exceptions import assertError
from ..image_utils.geometry import Rect
from ..image_utils.image import VLImage
from ..launch_options import LaunchOptions
from .base import (
    BaseDetection,
    ImageForDetection,
    ImageForRedetection,
    assertImageForDetection,
    getArgsForCoreDetectorForImages,
    getArgsForCoreRedetect,
    validateBatchDetectInput,
    validateReDetectInput,
)


def _createCoreBodies(image: ImageForRedetection) -> List[Human]:
    """
    Create core humans for redetection
    Args:
        image: image and bounding boxes for redetection

    Returns:
        Human object list. one object for one bbox
    """
    humans = [Human() for _ in range(len(image.bBoxes))]
    for index, human in enumerate(humans):
        human.img = image.image.coreImage
        human.detection.setRawRect(image.bBoxes[index].coreRectF)
        human.detection.setScore(1)
    return humans


class BodyDetection(BaseDetection):
    """Body detection class"""


def createHumanDetection(image: VLImage, detection: Detection):
    """
    Create human detection structure from core detection result.

    Args:
        image: origin image
        detection: sdk human detection

    Returns:
        HumanDetection structure
    """
    human = Human()
    human.img = image.coreImage
    human.detection = detection
    return BodyDetection(human, image)


def collectDetectionsResult(
    fsdkDetectRes,
    images: Union[List[Union[VLImage, ImageForDetection]], List[ImageForRedetection]],
    isRedectResult: bool = False,
) -> Union[List[List[BodyDetection]], List[List[Optional[BodyDetection]]]]:
    """
    Collect detection results from core reply and prepare human detections
    Args:
        fsdkDetectRes: fsdk (re)detect results
        images: incoming images
        isRedectResult: is redetect result or not
    Returns:
        return list of lists detection, order of detection lists is corresponding to order input images
    Raises:
        RuntimeError: if any detection is not valid and  processing detect ressult
    """
    res = []
    for imageIdx in range(fsdkDetectRes.getSize()):
        imagesDetections = []
        detections = fsdkDetectRes.getDetections(imageIdx)
        image = images[imageIdx]
        vlImage = image if isinstance(image, VLImage) else image.image
        for detectionIdx, detection in enumerate(detections):
            if detection.isValid():
                humanDetection = createHumanDetection(vlImage, detection)
            else:
                if not isRedectResult:
                    raise RuntimeError("invalid detection")
                humanDetection = None
            imagesDetections.append(humanDetection)
        res.append(imagesDetections)
    return res


def postProcessingOne(error: FSDKErrorResult, detectRes, image: VLImage) -> Optional[BodyDetection]:
    """
    Convert a core human detection to `BodyDetection` after detect one and error check.

    Args:
        error: detection error, usually error.isError is False
        detectRes: detections
        image: original image

    Raises:
        LunaSDKException: if detect is failed
        RuntimeError: if any detection is not valid
    Returns:
        human detection
    """
    assertError(error)
    detectionResCount = detectRes.getSize()
    if detectionResCount > 1 or detectionResCount == 0:
        raise RuntimeError(f"unexpected detection result count {detectionResCount}")

    detections = detectRes.getDetections(0)

    if len(detections) == 0:
        # human body not found
        return None
    detection = detections[0]
    humanDetection = createHumanDetection(image, detection)
    return humanDetection


def postProcessingRedetectOne(error: FSDKErrorResult, detectRes, image: VLImage) -> Optional[BodyDetection]:
    """
    Convert a core human detection to `BodyDetection` after redect and error check.

    Args:
        error: detection error, usually error.isError is False
        detectRes: detections
        image: original image

    Raises:
        LunaSDKException: if detect is failed
    Returns:
        face detection if  detection is valid (human body was found) otherwise None (human body was not found)
    """
    assertError(error)
    if detectRes.isValid():
        return BodyDetection(detectRes, image)
    return None


def postProcessing(
    error: FSDKErrorResult, fsdkDetectRes, images: List[Union[VLImage, ImageForDetection]]
) -> List[List[BodyDetection]]:
    """
    Convert core human detections from detector results to `HumanDetection` and error check.

    Args:
        error: detection error, usually error.isError is False
        fsdkDetectRes: core detection batch
        images: original images

    Returns:
        list, each item is face detections on corresponding image
    """
    assertError(error)
    return collectDetectionsResult(fsdkDetectRes, images=images, isRedectResult=False)  # type: ignore


def postProcessingRedect(
    error: FSDKErrorResult, fsdkDetectRes, images: List[ImageForRedetection]
) -> List[List[Optional[BodyDetection]]]:
    """
    Convert core human redetections from detector results to `HumanDetection` and error check.

    Args:
        error: detection error, usually error.isError is False
        fsdkDetectRes: core detection batch
        images: original images

    Returns:
        list, each item is face detections on corresponding image
    """
    assertError(error)
    return collectDetectionsResult(fsdkDetectRes, images=images, isRedectResult=True)  # type: ignore


# alias for detect one result
DetectOneResult = Union[None, BodyDetection]
# alias for detection result
DetectResult = List[List[BodyDetection]]
# alias for redection result
RedetectBatchResult = List[List[Union[BodyDetection, None]]]
# alias for redect one result
RedetectResult = Union[None, BodyDetection]


class BodyDetector:
    """
    Human body detector.

    Attributes:
        _detector (IDetectorPtr): core detector
        _launchOptions (LaunchOptions): detector launch options
    """

    __slots__ = (
        "_detector",
        "_launchOptions",
    )

    def __init__(self, detectorPtr, launchOptions: LaunchOptions):
        self._detector = detectorPtr
        self._launchOptions = launchOptions

    @staticmethod
    def _getDetectionType() -> HumanDetectionType:
        """
        Get  core detection type

        Returns:
            detection type
        """
        return HumanDetectionType.HDT_BOX

    @overload  # type: ignore
    def detectOne(
        self,
        image: VLImage,
        detectArea: Optional[Rect] = None,
        limit: int = 5,
        asyncEstimate: Literal[False] = False,
    ) -> DetectOneResult: ...

    @overload
    def detectOne(
        self,
        image: VLImage,
        detectArea: Optional[Rect],
        limit: int,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[DetectOneResult]: ...

    def detectOne(
        self,
        image: VLImage,
        detectArea: Optional[Rect] = None,
        limit: int = 5,
        asyncEstimate: bool = False,
    ):
        """
        Detect just one best detection on the image.

        Args:
            image: image. Format must be R8G8B8
            detectArea: rectangle area which contains human to detect. If not set will be set image.rect
            limit: max number of detections for input image
            asyncEstimate: estimate or run estimation in background
        Returns:
            human detection if human is found otherwise None
        Raises:
            LunaSDKException: if detectOne is failed or image format has wrong the format
        """
        assertImageForDetection(image)
        detectionType = self._getDetectionType()

        if detectArea is None:
            forDetection = ImageForDetection(image=image, detectArea=image.rect)
        else:
            forDetection = ImageForDetection(image=image, detectArea=detectArea)
        imgs, detectAreas = getArgsForCoreDetectorForImages([forDetection])
        if asyncEstimate:
            task = self._detector.asyncDetect(imgs, detectAreas, limit, detectionType)
            return AsyncTask(task, postProcessing=partial(postProcessingOne, image=image))
        error, detectRes = self._detector.detect(imgs, detectAreas, limit, detectionType)
        return postProcessingOne(error, detectRes, image)

    @overload
    def detect(
        self,
        images: List[Union[VLImage, ImageForDetection]],
        limit: int = 5,
        asyncEstimate: Literal[False] = False,
    ) -> Union[DetectResult, AsyncTask[DetectResult]]: ...

    @overload
    def detect(
        self,
        images: List[Union[VLImage, ImageForDetection]],
        limit: int,
        asyncEstimate: Literal[True] = True,
    ) -> Union[DetectResult, AsyncTask[DetectResult]]: ...

    def detect(
        self,
        images: List[Union[VLImage, ImageForDetection]],
        limit: int = 5,
        asyncEstimate=False,
    ):
        """
        Batch detect human bodies on images.

        Args:
            images: input images list. Format must be R8G8B8
            limit: max number of detections per input image
            asyncEstimate: estimate or run estimation in background
        Returns:
            asyncEstimate is False: return list of lists detection, order of detection lists is corresponding
                                    to order input images
            asyncEstimate isTrue:  async task
        """
        coreImages, detectAreas = getArgsForCoreDetectorForImages(images)
        detectionType = self._getDetectionType()
        validateBatchDetectInput(self._detector, coreImages, detectAreas)
        if asyncEstimate:
            task = self._detector.asyncDetect(coreImages, detectAreas, limit, detectionType)
            return AsyncTask(task, partial(postProcessing, images=images))
        error, fsdkDetectRes = self._detector.detect(coreImages, detectAreas, limit, detectionType)
        return postProcessing(error, fsdkDetectRes, images)

    def redetectOne(  # noqa: F811
        self,
        image: VLImage,
        bBox: Union[Rect, BodyDetection],
        asyncEstimate=False,
    ) -> Union[RedetectResult, AsyncTask[RedetectResult]]:
        """
        Redetect human body on an image in area, restricted with image.bBox, bBox or detection.

        Args:
            image: image with a bounding box, or just VLImage. If VLImage provided, one of bBox or detection
                should be defined.
            bBox: detection bounding box

        Returns:
            detection if human body found otherwise None if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException if an error occurs
        """
        assertImageForDetection(image)
        if isinstance(bBox, Rect):
            coreBBox = Detection(bBox.coreRectF, image.coreImage.getRect(), 1.0)
        else:
            coreBBox = bBox.coreEstimation.detection

        validateReDetectInput(self._detector, image.coreImage, coreBBox)
        if asyncEstimate:
            task = self._detector.asyncRedetectOne(image.coreImage, coreBBox, self._getDetectionType())
            return AsyncTask(task, partial(postProcessingRedetectOne, image=image))
        error, detectRes = self._detector.redetectOne(image.coreImage, coreBBox, self._getDetectionType())
        return postProcessingRedetectOne(error, detectRes, image)

    def redetect(
        self,
        images: List[ImageForRedetection],
        asyncEstimate=False,
    ) -> Union[RedetectBatchResult, AsyncTask[RedetectBatchResult]]:
        """
        Redetect human on each image.image in area, restricted with image.bBox.

        Args:
            images: images with a bounding boxes,
            asyncEstimate: estimate or run estimation in background

        Returns:
            detections if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException if an error occurs, context contains all errors
        """
        coreImages, detectAreas = getArgsForCoreRedetect(images)
        validateReDetectInput(self._detector, coreImages, detectAreas)
        if asyncEstimate:
            task = self._detector.asyncRedetect(coreImages, detectAreas, self._getDetectionType())
            pProcessing = partial(postProcessingRedect, images=images)
            return AsyncTask(task, pProcessing)
        error, fsdkDetectRes = self._detector.redetect(coreImages, detectAreas, self._getDetectionType())
        return postProcessingRedect(error, fsdkDetectRes, images=images)
