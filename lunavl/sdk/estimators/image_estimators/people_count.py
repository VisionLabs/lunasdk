from enum import Enum
from typing import List, Literal, NamedTuple, Tuple, Union, overload

import FaceEngine
from FaceEngine import CrowdEstimation, FSDKErrorResult

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.base import BaseEstimation, Landmarks
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator
from lunavl.sdk.image_utils.geometry import CoreRectI, Point, Rect, Vec2D
from lunavl.sdk.image_utils.image import CoreImage, VLImage


class EstimationTargets(Enum):
    """PeopleCount estimation targets"""

    # return people count and coordinates
    T1 = FaceEngine.CrowdRequest.estimateHeadCountAndCoords
    # return only people count
    T2 = FaceEngine.CrowdRequest.estimateHeadCount


class PeopleCount(BaseEstimation):

    @property
    def count(self):
        return self._coreEstimation.count

    @property
    def coordinates(self) -> Tuple[Vec2D, ...]:
        return tuple(Vec2D(point.x, point.y) for point in self._coreEstimation.points.getPoints())

    def asDict(self) -> dict:
        return {"count": self.count, "coordinates": self.coordinates}


class ImageForPeopleEstimation(NamedTuple):
    """
    Structure for transfer image and detect area to people estimator

    Attributes:
        image: image for detection
        detectArea: area for people detection
    """

    image: VLImage
    detectArea: Rect


def getEstimatorArgsFromImages(
    images: List[Union[VLImage, ImageForPeopleEstimation, Tuple[VLImage, Rect]]]
) -> Tuple[List[CoreImage], List[CoreRectI]]:
    """
    Create args for people estimation from image list
    Args:
        images: list of images for estimation

    Returns:
        tuple: first - list core images
               second - detect area for corresponding images
    """

    coreImages, detectAreas = [], []

    for image in images:
        if isinstance(image, VLImage):
            img = image
            detectAreas.append(image.coreImage.getRect())
        else:
            img = image[0]
            detectAreas.append(image[1].coreRectI)
        coreImages.append(img.coreImage)

    return coreImages, detectAreas


def postProcessingBatchV1(error: FSDKErrorResult, crowdEstimations: List[CrowdEstimation]) -> List[int]:
    """
    Post processing batch people count estimation

    Args:
        error: estimation error
        crowdEstimations: list of people count estimations

    Returns:
        list of people quantities
    """
    assertError(error)
    return [estimation.count for estimation in crowdEstimations]


def postProcessingV1(error: FSDKErrorResult, crowdEstimation: CrowdEstimation) -> int:
    """
    Post processing single people count estimation

    Args:
        error: estimation error
        crowdEstimations: people count estimation

    Returns:
        people count
    """
    assertError(error)
    return crowdEstimation[0].count


def postProcessingBatchV2(
    error: FSDKErrorResult, crowdEstimations: List[CrowdEstimation]
) -> List[PeopleCount]:
    """
    Post processing batch people count estimation

    Args:
        error: estimation error
        crowdEstimations: list of people count estimations

    Returns:
        list of people quantities
    """
    assertError(error)
    return [PeopleCount(estimation) for estimation in crowdEstimations]


def postProcessingV2(error: FSDKErrorResult, crowdEstimation: CrowdEstimation) -> PeopleCount:
    """
    Post processing single people count estimation

    Args:
        error: estimation error
        crowdEstimations: people count estimation

    Returns:
        people count
    """
    assertError(error)
    return PeopleCount(crowdEstimation[0])


class PeopleCountEstimatorV2(BaseEstimator):
    """People count estimator. Estimate people count feature and coordinates"""

    @overload  # type: ignore
    def estimate(
        self,
        image: Union[VLImage, ImageForPeopleEstimation],
        estimationTargets: EstimationTargets,
        asyncEstimate: Literal[False] = False,
    ) -> PeopleCount: ...

    @overload
    def estimate(
        self,
        image: Union[VLImage, ImageForPeopleEstimation],
        estimationTargets: EstimationTargets,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[PeopleCount]: ...

    def estimate(
        self,
        image: Union[VLImage, ImageForPeopleEstimation],
        estimationTargets: EstimationTargets = EstimationTargets.T1,
        asyncEstimate: bool = False,
    ):
        """
        Estimate people count from single image

        Args:
            image: vl image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated people count or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        if isinstance(image, VLImage):
            detectArea = image.coreImage.getRect()
        else:
            detectArea = image[1].coreRectI
            image = image[0]
        targets = estimationTargets.value
        validateInputByBatchEstimator(self._coreEstimator, [image.coreImage], [detectArea], targets)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate([image.coreImage], [detectArea], targets)
            return AsyncTask(task, postProcessingV2)
        error, crowdEstimation = self._coreEstimator.estimate([image.coreImage], [detectArea], targets)
        return postProcessingV2(error, crowdEstimation)

    @overload  # type: ignore
    def estimateBatch(
        self,
        images: List[Union[VLImage, ImageForPeopleEstimation, Tuple[VLImage, Rect]]],
        estimationTargets: EstimationTargets,
        asyncEstimate: Literal[False] = False,
    ) -> List[PeopleCount]: ...

    @overload
    def estimateBatch(
        self,
        images: List[Union[VLImage, ImageForPeopleEstimation, Tuple[VLImage, Rect]]],
        estimationTargets: EstimationTargets,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[List[PeopleCount]]: ...

    def estimateBatch(
        self,
        images: List[Union[VLImage, ImageForPeopleEstimation, Tuple[VLImage, Rect]]],
        estimationTargets: EstimationTargets = EstimationTargets.T1,
        asyncEstimate: bool = False,
    ):
        """
        Estimate people count from single image

        Args:
            images: list of vl image
            asyncEstimate: estimate or run estimation in background
            estimationTargets: targets to return

        Returns:
            list of estimated people count or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImages, detectAreas = getEstimatorArgsFromImages(images)
        targets = estimationTargets.value
        validateInputByBatchEstimator(self._coreEstimator, coreImages, detectAreas, targets)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages, detectAreas, targets)
            return AsyncTask(task, postProcessingBatchV2)
        error, crowdEstimations = self._coreEstimator.estimate(coreImages, detectAreas, targets)
        return postProcessingBatchV2(error, crowdEstimations)


class PeopleCountEstimatorV1(BaseEstimator):
    """People count estimator. Estimate only people count feature. Deprecated"""

    @overload  # type: ignore
    def estimate(
        self,
        image: Union[VLImage, ImageForPeopleEstimation],
        asyncEstimate: Literal[False] = False,
    ) -> int: ...

    @overload
    def estimate(
        self,
        image: Union[VLImage, ImageForPeopleEstimation],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[int]: ...

    def estimate(
        self,
        image: Union[VLImage, ImageForPeopleEstimation],
        asyncEstimate: bool = False,
    ):
        """
        Estimate people count from single image

        Args:
            image: vl image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated people count or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        estimator = PeopleCountEstimatorV2(self._coreEstimator, launchOptions=self.launchOptions)
        if asyncEstimate:
            task = estimator.estimate(image, EstimationTargets.T2, True)
            task.postProcessing = postProcessingV1  # type: ignore
            return task
        else:
            return estimator.estimate(image, EstimationTargets.T2, False).count

    @overload  # type: ignore
    def estimateBatch(
        self,
        images: List[Union[VLImage, ImageForPeopleEstimation, Tuple[VLImage, Rect]]],
        asyncEstimate: Literal[False] = False,
    ) -> List[int]: ...

    @overload
    def estimateBatch(
        self,
        images: List[Union[VLImage, ImageForPeopleEstimation, Tuple[VLImage, Rect]]],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[List[int]]: ...

    def estimateBatch(
        self,
        images: List[Union[VLImage, ImageForPeopleEstimation, Tuple[VLImage, Rect]]],
        asyncEstimate: bool = False,
    ):
        """
        Estimate people count from single image

        Args:
            images: list of vl image
            asyncEstimate: estimate or run estimation in background

        Returns:
            list of estimated people count or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        estimator = PeopleCountEstimatorV2(self._coreEstimator, launchOptions=self.launchOptions)
        if asyncEstimate:
            task = estimator.estimateBatch(images, EstimationTargets.T2, True)
            task.postProcessing = postProcessingBatchV1  # type: ignore
            return task
        else:
            res = estimator.estimateBatch(images, EstimationTargets.T2, False)
            return [estimation.count for estimation in res]


PeopleCountEstimator = PeopleCountEstimatorV1
