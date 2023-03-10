from typing import List, Union

from FaceEngine import DynamicRangeEstimation, FSDKErrorResult

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.estimators.base import BaseEstimator, ImageWithFaceDetection
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator


def postProcessingBatch(error: FSDKErrorResult, dynamicRangeEstimations: List[DynamicRangeEstimation]) -> List[float]:
    """
    Post processing batch dynamic range estimation

    Args:
        error: estimation error
        dynamicRangeEstimations: list of dynamic range estimations

    Returns:
        list of dynamic range
    """
    assertError(error)
    return [estimation.dynamicRangeScore for estimation in dynamicRangeEstimations]


def postProcessing(error: FSDKErrorResult, dynamicRangeEstimator: DynamicRangeEstimation) -> float:
    """
    Post processing single dynamic range estimation

    Args:
        error: estimation error
        dynamicRangeEstimator: dynamic range estimation

    Returns:
        dynamic range
    """
    assertError(error)
    return dynamicRangeEstimator[0].dynamicRangeScore


class DynamicRangeEstimator(BaseEstimator):
    """Dynamic range estimator"""

    def estimate(  # type: ignore
        self,
        imageWithFaceDetection: Union[ImageWithFaceDetection, FaceDetection],
        asyncEstimate: bool = False,
    ):
        """
        Estimate dynamic range from single image

        Args:
            imageWithFaceDetection: image with face detection
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated dynamic range or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        if isinstance(imageWithFaceDetection, FaceDetection):
            coreImage = imageWithFaceDetection.image.coreImage
            bbox = imageWithFaceDetection.boundingBox.coreEstimation
        else:
            coreImage = imageWithFaceDetection[0].coreImage
            bbox = imageWithFaceDetection[1].coreEstimation
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate([coreImage], [bbox])
            return AsyncTask(task, postProcessing)
        error, dynamicRangeEstimation = self._coreEstimator.estimate([coreImage], [bbox])
        return postProcessing(error, dynamicRangeEstimation)

    def estimateBatch(
        self,
        batch: List[Union[ImageWithFaceDetection, FaceDetection]],
        asyncEstimate: bool = False,
    ):
        """
        Estimate dynamic range from single image

        Args:
            batch: list of image with face detection or face detections
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated dynamic range or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImages = []
        detections = []
        for data in batch:
            if isinstance(data, FaceDetection):
                coreImages.append(data.image.coreImage)
                detections.append(data.boundingBox.coreEstimation)
            else:
                coreImages.append(data[0].coreImage)
                detections.append(data[1].coreEstimation)
        validateInputByBatchEstimator(self._coreEstimator, coreImages, detections)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages, detections)
            return AsyncTask(task, postProcessingBatch)
        error, coreDynamicRangeList = self._coreEstimator.estimate(coreImages, detections)
        return postProcessingBatch(error, coreDynamicRangeList)
