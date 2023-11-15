"""Module contains an approximate garbage score estimator

See ags_.
"""
from typing import List, Optional, Union, overload, Literal

from FaceEngine import FSDKErrorResult  # pylint: disable=E0611,E0401

from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.errors.exceptions import assertError

from ..base import BaseEstimator, ImageWithFaceDetection
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ...async_task import AsyncTask, DefaultPostprocessingFactory


def postProcessing(
    error: FSDKErrorResult,
    estimation: float,
) -> float:
    """
    Postprocessing single core estimation
    Args:
        error: estimation error
        estimation: core estimation

    Returns:
        estimation
    """
    assertError(error)
    return estimation


def postProcessingBatch(
    error: FSDKErrorResult,
    estimations: List[float],
) -> List[float]:
    """
    Postprocessing batch core estimations
    Args:
        error: estimation error
        estimations: core estimations

    Returns:
        list of estimations
    """
    assertError(error)
    return estimations


class AGSEstimator(BaseEstimator):
    """
    Approximate garbage score estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        detection: Optional[FaceDetection] = None,
        imageWithFaceDetection: ImageWithFaceDetection = None,
        asyncEstimate: Literal[False] = False,
    ) -> float:
        ...

    @overload
    def estimate(
        self,
        detection: Optional[FaceDetection],
        imageWithFaceDetection: ImageWithFaceDetection,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[float]:
        ...

    def estimate(
        self,
        detection: Optional[FaceDetection] = None,
        imageWithFaceDetection: Optional[ImageWithFaceDetection] = None,
        asyncEstimate: bool = False,
    ) -> Union[float, AsyncTask[float]]:
        """
        Estimate ags for single image/detection.

        Args:
            detection: face detection
            imageWithFaceDetection: image with face detection
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated ags, float in range[0,1] if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if image and detection are None
        """
        if detection is None:
            if imageWithFaceDetection is None:
                raise ValueError("image and boundingBox or detection must be not None")
            coreImage = imageWithFaceDetection.image.coreImage
            bbox = imageWithFaceDetection.boundingBox.coreEstimation
        else:
            coreImage = detection.image.coreImage
            bbox = detection.boundingBox.coreEstimation
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImage, bbox)
            return AsyncTask(task, postProcessing)
        error, estimations = self._coreEstimator.estimate(coreImage, bbox)
        return postProcessing(error, estimations)

    @overload  # type: ignore
    def estimateBatch(
        self,
        detections: Union[List[FaceDetection], List[ImageWithFaceDetection]],
        asyncEstimate: Literal[False] = False,
    ) -> List[float]:
        ...

    @overload
    def estimateBatch(
        self, detections: Union[List[FaceDetection], List[ImageWithFaceDetection]], asyncEstimate: Literal[True]
    ) -> AsyncTask[List[float]]:
        ...

    def estimateBatch(
        self,
        detections: Union[List[FaceDetection], List[ImageWithFaceDetection]],
        asyncEstimate: bool = False,
    ) -> Union[List[float], AsyncTask[List[float]]]:
        """
        Estimate ags for list of detections.

        Args:
            detections: face detection list or list of image with its face detection
            asyncEstimate: estimate or run estimation in background

        Returns:
            list of estimated ags, float in range[0,1] if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if empty image list and empty detection list or images count not match bounding boxes count
        """
        coreImages = [detection.image.coreImage for detection in detections]
        boundingBoxEstimations = [detection.boundingBox.coreEstimation for detection in detections]

        validateInputByBatchEstimator(self._coreEstimator, coreImages, boundingBoxEstimations)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages, boundingBoxEstimations)
            return AsyncTask(task, postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(coreImages, boundingBoxEstimations)
        return postProcessingBatch(error, estimations)
