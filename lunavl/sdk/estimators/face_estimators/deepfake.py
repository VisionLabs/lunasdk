"""
Module contains a  livenessv1 estimator.

See `livenessv1`_.
"""
from enum import Enum
from typing import List, Literal, Union, overload

from FaceEngine import (  # pylint: disable=E0611,E0401
    DeepFakeEstimation,
    DeepFakeEstimationState,
    IDeepFakeEstimatorPtr,
)
from lunavl.sdk.async_task import AsyncTask, DefaultPostprocessingFactory
from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator
from lunavl.sdk.faceengine.setting_provider import DeepFakeEstimationMode


class DeepFakeState(Enum):
    """
    Deepfake state enum
    """

    # no mask on the face
    Real = "real"  # real human
    Fake = "fake"  # spoof

    @staticmethod
    def fromCoreEmotion(coreState: DeepFakeEstimationState) -> "DeepFakeState":
        """
        Get enum element by core deepfake state.

        Args:
            coreState: enum value from core

        Returns:
            corresponding state
        """
        if coreState == DeepFakeEstimationState.Real:
            return DeepFakeState.Real
        return DeepFakeState.Fake


class DeepFake(BaseEstimation):
    """
    Liveness structure (LivenessOneShotRGBEstimation).

    Attributes:
        prediction: liveness prediction

    Estimation properties:

        - score
        - quality
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: DeepFakeEstimation):
        """
        Init.

        Args:
            coreEstimation: core estimation
        """
        super().__init__(coreEstimation)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            {"prediction": self.prediction, "estimations": {"quality": self.quality, "score": self.score}}
        """
        return {"state": self.coreEstimation.State.name.lower(), "score": self.coreEstimation.score}

    @property
    def score(self) -> float:
        """
        Liveness score

        Returns:
            liveness score
        """
        return self._coreEstimation.score

    @property
    def state(self) -> DeepFakeState:
        """
        Liveness quality score

        Returns:
            liveness quality score
        """
        return DeepFakeState.fromCoreEmotion(self.coreEstimation.State)


POST_PROCESSING = DefaultPostprocessingFactory(DeepFake)


class DeepFakeEstimator(BaseEstimator):
    """
    Deep fake estimator.
    """

    def __init__(self, *args, mode: DeepFakeEstimationMode = DeepFakeEstimationMode.Default, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = mode

    @property
    def mode(self) -> DeepFakeEstimationMode:
        """Estimation mode getter"""
        return self._mode

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[False] = False,
    ) -> DeepFake:
        ...

    @overload
    def estimate(
        self,
        faceDetection: FaceDetection,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[DeepFake]:
        ...

    def estimate(  # type: ignore
        self,
        faceDetection: FaceDetection,
        asyncEstimate: bool = False,
    ) -> Union[DeepFake, AsyncTask[DeepFake]]:
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
    ) -> Union[List[DeepFake], AsyncTask[List[DeepFake]]]:
        """
        Batch estimate deep fake feature

        Args:
            faceDetections: face detection list
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated liveness if asyncEstimate is False otherwise async task
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
