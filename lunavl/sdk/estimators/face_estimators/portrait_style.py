"""
Module contains portrait style estimator.

See `portrait_style`_.
TODO
"""
from typing import Dict, List, Union

import FaceEngine

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.detectors.facedetector import FaceDetection

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ..base import BaseEstimator, ImageWithFaceDetection
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator


class PortraitStyle(BaseEstimation):
    """
    PortraitStyle. Estimation of portrait style on face detection.

    Estimation properties:

        - status
        - nonPortrait
        - portrait
        - hiddenShoulders
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: FaceEngine.PortraitStyleEstimation):
        """
        Init.

        Args:
            coreEstimation: core portrait style estimation.
        """

        super().__init__(coreEstimation)

    @property
    def nonPortrait(self) -> float:
        """Not a portrait score [0, 1.0]."""
        return self._coreEstimation.nonPortraitScore

    @property
    def portrait(self) -> float:
        """Portrait score [0, 1.0]."""
        return self._coreEstimation.portraitScore

    @property
    def hiddenShoulders(self) -> float:
        """Hidden shoulds score [0, 1.0]."""
        return self._coreEstimation.hiddenShouldersScore

    @property
    def status(self) -> bool:
        """
        Prediction status.
        Returns:
            True if portait was estimated otherwise false
        """
        return self._coreEstimation.status == FaceEngine.PortraitStyleStatus.Portrait

    def asDict(self) -> Dict[str, Union[float, bool]]:
        """Convert estimation to dict."""
        return {"non_portrait": self.nonPortrait, "portrait": self.portrait, "hidden_shoulders": self.hiddenShoulders, "status": self.status}


POST_PROCESSING = DefaultPostprocessingFactory(PortraitStyle)


class PortraitStyleEstimator(BaseEstimator):
    """
    Estimate whether shoulders are visible and parallel.
    """
    def estimate(  # type: ignore
        self,
        faceDetection: FaceDetection,
        asyncEstimate: bool = False,
    ) -> Union[PortraitStyle, AsyncTask[PortraitStyle]]:
        """
        Estimate portrait style from single image

        Args:
            imageWithFaceDetection: image with face detection
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated portraite style or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImage = faceDetection.image.coreImage
        detection = faceDetection.coreEstimation.detection
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImage, detection)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, portraitStyle = self._coreEstimator.estimate(coreImage, detection)
        return POST_PROCESSING.postProcessing(error, portraitStyle)

    def estimateBatch(  # type: ignore
        self,
        faceDetections: List[FaceDetection],
        qualityThreshold: float | None = None,
        asyncEstimate: bool = False,
    ) -> Union[List[PortraitStyle], AsyncTask[List[PortraitStyle]]]:
        """
        Estimate portrait style from chunk of images.

        Args:
            batch: list of image with face detection or face detections
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated dynamic range or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImages = [x.image.coreImage for x in faceDetections]
        detections = [x.coreEstimation.detection for x in faceDetections]
        validateInputByBatchEstimator(self._coreEstimator, coreImages, detections)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages, detections)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, coreDynamicRangeList = self._coreEstimator.estimate(coreImages, detections)
        return POST_PROCESSING.postProcessingBatch(error, coreDynamicRangeList)
