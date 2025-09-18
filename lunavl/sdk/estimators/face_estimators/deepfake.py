"""
Module contains a  deepfake estimator.

See `deepfake`_.
"""

from enum import Enum
from typing import List, Literal, Union, overload

from FaceEngine import DeepFakeEstimation, DeepFakeEstimationState, DeepFakeMode  # pylint: disable=E0611,E0401

from lunavl.sdk.async_task import AsyncTask, DefaultPostprocessingFactory
from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator


class DeepfakeEstimationMode(Enum):
    """
    Deepfake estimation mode
    """

    Default = 0
    M2 = 2

    @property
    def coreEstimatorType(self) -> DeepFakeMode:
        return DeepFakeMode(self.value)


class DeepfakePrediction(Enum):
    """
    Deepfake state enum
    """

    # no mask on the face
    Real = "real"  # real human
    Fake = "fake"  # spoof

    @staticmethod
    def fromCoreDeepfake(coreState: DeepFakeEstimationState) -> "DeepfakePrediction":
        """
        Get enum element by core deepfake state.

        Args:
            coreState: enum value from core

        Returns:
            corresponding state
        """
        if coreState == DeepFakeEstimationState.Real:
            return DeepfakePrediction.Real
        return DeepfakePrediction.Fake


class Deepfake(BaseEstimation):
    """
    Deepfake structure

    Attributes:
        prediction: deepfake prediction

    Estimation properties:

        - score
        - quality
    """

    __slots__ = ("prediction",)
    #  pylint: disable=W0235

    def __init__(self, coreEstimation: DeepFakeEstimation):
        """
        Init.

        Args:
            coreEstimation: core estimation
        """
        super().__init__(coreEstimation)
        self.prediction = DeepfakePrediction.fromCoreDeepfake(coreEstimation.State)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            {"state": self.state, "estimations": {"score": self.score}}
        """
        return {"prediction": self.prediction.value, "score": self.coreEstimation.score}

    @property
    def score(self) -> float:
        """
        Deepfake score

        Returns:
            deepfake score, higher better
        """
        return self._coreEstimation.score


POST_PROCESSING = DefaultPostprocessingFactory(Deepfake)


class DeepfakeEstimator(BaseEstimator):
    """
    Deep fake estimator.
    """

    def __init__(self, *args, mode: DeepfakeEstimationMode, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = mode

    @property
    def mode(self) -> DeepfakeEstimationMode:
        """Estimation mode getter"""
        return self._mode

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[False] = False,
    ) -> Deepfake: ...

    @overload
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[Deepfake]: ...

    def estimate(  # type: ignore
        self,
        faceDetection: FaceDetection,
        asyncEstimate: bool = False,
    ) -> Union[Deepfake, AsyncTask[Deepfake]]:
        """
        Estimate a deep fake

        Args:
            faceDetection: face detection
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated deep fake if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(
                faceDetection.image.coreImage,
                faceDetection.coreEstimation.detection,
            )
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, estimation = self._coreEstimator.estimate(
            faceDetection.image.coreImage,
            faceDetection.coreEstimation.detection,
        )
        return POST_PROCESSING.postProcessing(error, estimation)

    #  pylint: disable=W0221
    def estimateBatch(  # type: ignore
        self,
        faceDetections: List[FaceDetection],
        asyncEstimate: bool = False,
    ) -> Union[List[Deepfake], AsyncTask[List[Deepfake]]]:
        """
        Batch estimate deep fake feature

        Args:
            faceDetections: face detection list
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated deepfake feature if asyncEstimate is False otherwise async task
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
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(
            coreImages,
            detections,
        )
        return POST_PROCESSING.postProcessingBatch(error, estimations)
