"""
Module contains an image modification estimator.

See `image modification_.
"""

from enum import Enum
from typing import List, Union

from FaceEngine import FSDKErrorResult, ImageModificationStatus

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator
from lunavl.sdk.image_utils.image import VLImage


class ModificationStatus(Enum):
    """
    ModificationStatus enum
    """

    #: Unmodified
    Unmodified = 0
    #: Modified
    Modified = 1

    @classmethod
    def fromCoreModification(cls, coreEmotion) -> "ModificationStatus":
        """
        Get enum element by core modification.

        Args:
            coreEmotion:

        Returns:
            corresponding modification
        """
        return getattr(cls, coreEmotion.status.name)


class ImageModification(BaseEstimation):
    """
    Image modification estimation
    """

    @property
    def status(self) -> ModificationStatus:
        """ModificationStatus status getter"""
        return ModificationStatus.fromCoreModification(self._coreEstimation)

    @status.setter
    def status(self, status: ModificationStatus):
        """ModificationStatus status setter"""
        if status == ModificationStatus.Unmodified:
            self._coreEstimation.status = ImageModificationStatus.Unmodified
        else:
            self._coreEstimation.status = ImageModificationStatus.Modified

    @property
    def score(self) -> float:
        return self._coreEstimation.score

    def __repr__(self):
        return f"status: {self.status}, score: {self.score}"

    def asDict(self) -> dict:
        return {"score": self.score, "status": self.status.value}


def postProcessingBatch(error: FSDKErrorResult, estimations) -> list[ImageModification]:
    """
    Post processing batch image modification estimation
    Args:
        error:  estimation error
        estimations: estimated image modifications

    Returns:
        list of `ImageModification`
    """
    assertError(error)

    return [ImageModification(estimation) for estimation in estimations]


def postProcessing(error: FSDKErrorResult, estimation) -> ImageModification:
    """
    Postprocessing single core image modification estimation
    Args:
        error: estimation error
        estimation: core estimation

    Returns:
        image modication
    """
    assertError(error)

    return ImageModification(estimation)


class ImageModificationEstimator(BaseEstimator):
    """
    ImageModificationEstimator.
    """

    def estimate(  # type: ignore
        self, image: VLImage, asyncEstimate: bool = False
    ) -> Union[ImageModification, AsyncTask[ImageModification]]:
        """
        Estimate image modification from image.

        Args:
            image: vl image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated orientation mode
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImage = image.coreImage
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImage)
            return AsyncTask(task, postProcessing)
        error, coreOrientationType = self._coreEstimator.estimate(coreImage)
        return postProcessing(error, coreOrientationType)

    def estimateBatch(
        self, images: List[VLImage], asyncEstimate: bool = False
    ) -> Union[List[ImageModification], AsyncTask[List[ImageModification]]]:
        """
        Batch estimate image modification from images.

        Args:
            images: vl image or face warp list
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated orientation mode list if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImages = [img.coreImage for img in images]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, postProcessingBatch)
        error, coreOrientationTypeList = self._coreEstimator.estimate(coreImages)
        return postProcessingBatch(error, coreOrientationTypeList)
