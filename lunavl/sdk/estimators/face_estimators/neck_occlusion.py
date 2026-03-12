from enum import Enum
from typing import List, Literal, Union, overload

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ...base import BaseEstimation
from ...detectors.facedetector import FaceDetection
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator

_MAP_CORE_NAME = {"NeckOpen": "OpenNeck", "NeckCovered": "OccludedNeck", "NeckUnknown": "Unknown"}


class NeckOcclusionEnum(Enum):
    """Neck occlusion state enum"""

    OpenNeck = "open_neck"  # open neck
    OccludedNeck = "occluded_neck"  # occluded neck
    Unknown = "unknown"  # unknown state

    @classmethod
    def fromCoreNeckOcclusion(cls, coreNeckOcclusion) -> "NeckOcclusionEnum":
        """
        Get enum element by core neck occlusion state.

        Args:
            coreNeckOcclusion: core neck occlusion enum

        Returns:
            corresponding neck occlusion
        """
        return cls[_MAP_CORE_NAME[coreNeckOcclusion.name]]


class NeckOcclusion(BaseEstimation):
    """
    Class for neck occlusion estimation.

    Neck occlusion properties:

        - openNeck
        - occludedNeck
        - unknown
        - predominant
    """

    @property
    def openNeck(self) -> float:
        """
        Get `open_neck` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.openScore

    @property
    def occludedNeck(self):
        """
        Get `occluded_neck ` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.coveredScore

    @property
    def unknown(self):
        """
        Get `unknown` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.unknownScore

    @property
    def predominant(self) -> NeckOcclusionEnum:
        """Get neck occlusion predominant state"""
        return NeckOcclusionEnum.fromCoreNeckOcclusion(self._coreEstimation.result)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict in platform format
        """
        return {
            "predominant_state": NeckOcclusionEnum.fromCoreNeckOcclusion(self._coreEstimation.result).value,
            "estimations": {
                "open_neck": self.openNeck,
                "occluded_neck": self.occludedNeck,
                "unknown": self.unknown,
            },
        }


POST_PROCESSING = DefaultPostprocessingFactory(NeckOcclusion)
postProcessing = POST_PROCESSING.postProcessing
postProcessingBatch = POST_PROCESSING.postProcessingBatch


class NeckOcclusionEstimator(BaseEstimator):
    """
    Neck occlusion estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[False] = False,
    ) -> NeckOcclusion: ...

    @overload
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[NeckOcclusion]: ...

    def estimate(  # type: ignore
        self,
        faceDetection: FaceDetection,
        asyncEstimate: bool = False,
    ) -> Union[NeckOcclusion, AsyncTask[NeckOcclusion]]:
        """
        Estimate neck occlusion state

        Args:
            faceDetection: face detection
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated neck occlusion state if asyncEstimate is False otherwise async task
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
    ) -> Union[List[NeckOcclusion], AsyncTask[List[NeckOcclusion]]]:
        """
        Batch estimate neck occlusion state

        Args:
            faceDetections: face detection list
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated neck occlusion if asyncEstimate is False otherwise async task
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
