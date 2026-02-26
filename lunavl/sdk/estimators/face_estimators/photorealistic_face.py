from enum import Enum
from typing import List, Literal, Union, overload

from FaceEngine import PhotorealisticFaceEstimationType as CorePhotorealisticEnum  # pylint: disable=E0611,E0401

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ...base import BaseEstimation
from ...detectors.facedetector import FaceDetection
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator


class PhotorealisticEnum(Enum):
    """Photorealistic face prediction enum"""

    Drawn = "drawn"  # drawn face
    Real = "real "  # real fce

    @classmethod
    def fromCorePhotorealistic(cls, corePhotorealisticFace) -> "PhotorealisticEnum":
        """
        Get enum element by photorealistic face prediction enum.

        Args:
            corePhotorealisticFace: core photorealistic face enum

        Returns:
            corresponding photorealistic prediction
        """
        return cls[corePhotorealisticFace.name.lower()]


class PhotorealisticPrediction(BaseEstimation):
    """
    Class for photorealistic prediction estimation.

    Properties:
        - score
    """

    @property
    def score(self) -> float:
        """
        Get score estimation score. 0 means drawn , 1 - real

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.realScore

    @property
    def status(self) -> bool:
        """Get photorealistic face status"""
        return True if self._coreEstimation.result == CorePhotorealisticEnum.Real else False

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict in platform format
        """
        return {"score": self.score, "status": self.status}


POST_PROCESSING = DefaultPostprocessingFactory(PhotorealisticPrediction)
postProcessing = POST_PROCESSING.postProcessing
postProcessingBatch = POST_PROCESSING.postProcessingBatch


class PhotorealisticFaceEstimator(BaseEstimator):
    """
    Photorealistic face estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[False] = False,
    ) -> PhotorealisticPrediction: ...

    @overload
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[PhotorealisticPrediction]: ...

    def estimate(  # type: ignore
        self,
        faceDetection: FaceDetection,
        asyncEstimate: bool = False,
    ) -> Union[PhotorealisticPrediction, AsyncTask[PhotorealisticPrediction]]:
        """
        Estimate photorealistic face status

        Args:
            faceDetection: face detection
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated photorealistic face status if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(
                faceDetection.image.coreImage,
                faceDetection.coreEstimation.detection,
            )
            return AsyncTask(task, postProcessing)
        error, estimation = self._coreEstimator.estimate(
            faceDetection.image.coreImage,
            faceDetection.coreEstimation.detection,
        )
        return postProcessing(error, estimation)

    #  pylint: disable=W0221
    def estimateBatch(  # type: ignore
        self,
        faceDetections: List[FaceDetection],
        asyncEstimate: bool = False,
    ) -> Union[List[PhotorealisticPrediction], AsyncTask[List[PhotorealisticPrediction]]]:
        """
        Batch estimate photorealistic face status

        Args:
            faceDetections: face detection list
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated photorealistic face status if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [detection.image.coreImage for detection in faceDetections]
        detections = [detection.coreEstimation.detection for detection in faceDetections]

        validateInputByBatchEstimator(self._coreEstimator, coreImages, detections)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(
                coreImages,
                detections,
            )
            return AsyncTask(task, postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(
            coreImages,
            detections,
        )
        return postProcessingBatch(error, estimations)
