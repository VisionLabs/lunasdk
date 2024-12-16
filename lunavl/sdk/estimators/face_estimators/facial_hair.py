from enum import Enum
from typing import List, Literal, Union, overload

from FaceEngine import FacialHair as CoreFacialHairState, FacialHairEstimation  # pylint: disable=E0611,E0401

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
    def fromCoreFacialHairState(coreFacialHairState: CoreFacialHairState) -> "FacialHairState":
        """
        Get enum element by core emotion.

        Args:
            coreFacialHairState: an eye state form core

        Returns:
            corresponding eye state
        """
        return getattr(FacialHairState, coreFacialHairState.name)

    def __str__(self):
        return self.name.lower() if self.name != "NoHair" else "no_hair"


class FacialHair(BaseEstimation):
    """
    Facial hair estimation container

    Facial hair states:
        - beard
        - mustache
        - noFace
        - stubble
    """

    def __init__(self, coreFacialHair: FacialHairEstimation):
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
            beardScore - estimation of a beard on a face
            mustacheScore - estimation of a mustache on a face
            noHairScore - estimation of no hair on a face
            stubbleScore - estimation of a stubble on a face
        """
        print(self._coreEstimation.result)
        return {
            "predominant_facial_hair": self.predominateFacialHair,
            "estimations": {
                "beard": self.beardScore,
                "mustache": self.mustacheScore,
                "no_hair": self.noHairScore,
                "stubble": self.stubbleScore,
            },
        }

    @property
    def beardScore(self) -> float:
        """
        Get beard estimation value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.beardScore

    @property
    def mustacheScore(self) -> float:
        """
        Get mustache estimation value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.mustacheScore

    @property
    def noHairScore(self) -> float:
        """
        Get hairless estimation value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.noHairScore

    @property
    def stubbleScore(self) -> float:
        """
        Get stubble estimation value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.stubbleScore

    @property
    def predominateFacialHair(self) -> str:
        """
        Get predominate eyebrow expression (expression with max score value).

        Returns:
            eyebrow expression with max score value
        """
        result = FacialHairState.fromCoreFacialHairState(self._coreEstimation.result)
        return result.name.lower() if result.name != "NoHair" else "no_hair"


POST_PROCESSING = DefaultPostprocessingFactory(FacialHair)


class FacialHairEstimator(BaseEstimator):
    """
    Facial hair estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: Literal[False] = False) -> FacialHair: ...

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
        Estimate facial hair.

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
