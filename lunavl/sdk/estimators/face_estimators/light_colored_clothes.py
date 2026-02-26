from enum import Enum
from typing import List, Literal, Union, overload

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ...base import BaseEstimation
from ...detectors.facedetector import FaceDetection
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator

_MAP_CORE_NAME = {"Dark": "DarkClothes", "Light": "LightClothes", "Unknown": "Unknown"}


class LightColoredClothesEnum(Enum):
    """Light colored clothes enum"""

    DarkClothes = "dark_clothes"  # dark clothes
    LightClothes = "light_clothes"  # light clothes
    Unknown = "unknown"  # unknown state

    @classmethod
    def fromCoreLightColoredClothes(cls, coreLightColoredClothes) -> "LightColoredClothesEnum":
        """
        Get enum element by core lightness of clothing.

        Args:
            coreLightColoredClothes: core light colored clothes enum

        Returns:
            corresponding light colored clothes
        """
        return cls[_MAP_CORE_NAME[coreLightColoredClothes.name]]


class LightColoredClothes(BaseEstimation):
    """
    Class for lightness of clothing estimation.

    Lightness of clothing properties:

        - darkClothes
        - lightClothes
        - unknown
        - predominant
    """

    @property
    def darkClothes(self) -> float:
        """
        Get `dark_clothes` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.darkScore

    @property
    def lightClothes(self):
        """
        Get `light_clothes ` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.lightScore

    @property
    def unknown(self):
        """
        Get `unknown` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.unknownScore

    @property
    def predominant(self) -> LightColoredClothesEnum:
        """Get lightness of clothing predominant estimation"""
        return LightColoredClothesEnum.fromCoreLightColoredClothes(self._coreEstimation.result)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict in platform format
        """
        return {
            "predominant": self.predominant.value,
            "estimations": {
                "light_clothes": self.lightClothes,
                "dark_clothes": self.darkClothes,
                "unknown": self.unknown,
            },
        }


POST_PROCESSING = DefaultPostprocessingFactory(LightColoredClothes)
postProcessing = POST_PROCESSING.postProcessing
postProcessingBatch = POST_PROCESSING.postProcessingBatch


class LightColoredClothesEstimator(BaseEstimator):
    """
    Light colored clothes estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[False] = False,
    ) -> LightColoredClothes: ...

    @overload
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[LightColoredClothes]: ...

    def estimate(  # type: ignore
        self,
        faceDetection: FaceDetection,
        asyncEstimate: bool = False,
    ) -> Union[LightColoredClothes, AsyncTask[LightColoredClothes]]:
        """
        Estimate whether light colored clothes is being worn

        Args:
            faceDetection: face detection
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated liveness if asyncEstimate is False otherwise async task
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
    ) -> Union[List[LightColoredClothes], AsyncTask[List[LightColoredClothes]]]:
        """
        Batch estimate lightness of clothing

        Args:
            faceDetections: face detection list
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated lightness of clothing if asyncEstimate is False otherwise async task
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
