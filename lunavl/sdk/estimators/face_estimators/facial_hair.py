from enum import Enum
from typing import List, Literal, Union, overload

from FaceEngine import FaceOcclusionEstimation, FaceOcclusionState, FaceOcclusionType, State as CoreHairState  # pylint: disable=E0611,E0401

from lunavl.sdk.async_task import AsyncTask, DefaultPostprocessingFactory
from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarp, FaceWarpedImage


class FacialHairState(Enum):
    """
    Enum for eye states.
    """
    #: No hair on face
    NoHair = 0
    #: Stubble on face
    Stubble = 1
    #: Mustache on face
    Mustache = 2
    #: Beard on face
    Beard = 3

    @staticmethod
    def fromCoreFacialHairState(coreFacialHairState: CoreHairState) -> "FacialHairState":
        """
        Get enum element by core emotion.

        Args:
            coreFacialHairState: an eye state form core

        Returns:
            corresponding eye state
        """
        return getattr(FacialHairState, coreFacialHairState.name)


class FacialHair(BaseEstimation):
    """
    Facial hair estimation container

    Facial hair states:
        - beard
        - mustache
        - noFace
        - stubble
    """

    def __init__(self, coreFacialHair):
        """
        Init of facial hair

        Args:
            coreFacialHair: core hair estimation.
        """

        super().__init__(coreFacialHair)

    def asDict(self) -> dict:
        """
        Convert estimation to dict.

        Returns:
            dict with keys:
            prediction - common prediction face occluded or not
            estimations - all scores
            face_occlusion - occlusion prediction by each face areas separately

        """
        return {
            "beard": self.beard,
            "mustache": self.mustache,
            "noHair": self.noHair,
            "stubble": self.stubble
        }

    @property
    def beard(self) -> float:
        """
        Get beard estimation value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.beard

    @property
    def mustache(self) -> float:
        """
        Get mustache estimation value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.mustache

    @property
    def noHair(self) -> float:
        """
        Get hairless estimation value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.noHair

    @property
    def stubble(self) -> float:
        """
        Get stubble estimation value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.stubble


POST_PROCESSING = DefaultPostprocessingFactory(FacialHair)


class FacialHairEstimator(BaseEstimator):
    """
    Face occlusion estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: Literal[False] = False
    ) -> FacialHair: ...

    @overload
    def estimate(
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: Literal[True]
    ) -> AsyncTask[FacialHair]: ...

    def estimate(  # type: ignore
        self,
        warp: Union[FaceWarp, FaceWarpedImage],
        asyncEstimate: bool = False,
    ) -> Union[FacialHair, AsyncTask[FacialHair]]:
        """
        Estimate face occlusion.

        Args:
            warp: warped image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated states if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """

        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, estimation = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return POST_PROCESSING.postProcessing(error, estimation)

    # pylint: disable=W0221
    def estimateBatch(
        self, warps: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[FacialHair], AsyncTask[List[FacialHair]]]:
        """
        Batch estimate facial hair state on warps.

        Args:
            warps: list of warped images
            asyncEstimate: estimate or run estimation in background

        Returns:
            list of estimated states if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if warps count not equals landmarks count
        """
        coreImages = [warp.warpedImage.coreImage for warp in warps]
        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(coreImages)
        return POST_PROCESSING.postProcessingBatch(error, estimations)
