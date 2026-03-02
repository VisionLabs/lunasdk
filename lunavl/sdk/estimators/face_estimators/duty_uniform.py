from enum import Enum
from typing import List, Literal, Union, overload

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ...base import BaseEstimation
from ...detectors.facedetector import FaceDetection
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator

_MAP_CORE_NAME = {"ShoulderStraps": "DutyClothes", "NoShoulderStraps": "RegularClothes", "Unknown": "Unknown"}


class DutyUniformEnum(Enum):
    """Duty uniform enum"""

    DutyClothes = "duty_clothes"  # duty clothes
    RegularClothes = "regular_clothes"  # regular clothes
    Unknown = "unknown"  # unknown state

    @classmethod
    def fromCoreDutyUniform(cls, coreDutyUniform) -> "DutyUniformEnum":
        """
        Get enum element by core duty uniform.

        Args:
            coreDutyUniform: core duty uniform enum

        Returns:
            corresponding duty uniform
        """
        return cls[_MAP_CORE_NAME[coreDutyUniform.name]]


class DutyUniform(BaseEstimation):
    """
    Class for duty uniform estimation.

    Duty uniform properties:

        - dutyClothes
        - regularClothes
        - unknown
        - predominant
    """

    @property
    def dutyClothes(self) -> float:
        """
        Get `duty_clothes` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.shoulderStrapsScore

    @property
    def regularClothes(self):
        """
        Get `regular_clothes ` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.noShoulderStrapsScore

    @property
    def unknown(self):
        """
        Get `unknown` predict value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.unknownScore

    @property
    def predominant(self) -> DutyUniformEnum:
        """Get duty uniform predominant estimation"""
        return DutyUniformEnum.fromCoreDutyUniform(self._coreEstimation.result)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict in platform format
        """
        return {
            "predominant_state": self.predominant.value,
            "estimations": {
                "duty_clothes": self.dutyClothes,
                "regular_clothes": self.regularClothes,
                "unknown": self.unknown,
            },
        }


POST_PROCESSING = DefaultPostprocessingFactory(DutyUniform)
postProcessing = POST_PROCESSING.postProcessing
postProcessingBatch = POST_PROCESSING.postProcessingBatch


class DutyUniformEstimator(BaseEstimator):
    """
    Duty uniform estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[False] = False,
    ) -> DutyUniform: ...

    @overload
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[DutyUniform]: ...

    def estimate(  # type: ignore
        self,
        faceDetection: FaceDetection,
        asyncEstimate: bool = False,
    ) -> Union[DutyUniform, AsyncTask[DutyUniform]]:
        """
        Estimate whether duty uniform is being worn

        Args:
            faceDetection: face detection
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated duty uniform if asyncEstimate is False otherwise async task
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
    ) -> Union[List[DutyUniform], AsyncTask[List[DutyUniform]]]:
        """
        Batch estimate  whether duty uniform is being worn

        Args:
            faceDetections: face detection list
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated uniform if asyncEstimate is False otherwise async task
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
