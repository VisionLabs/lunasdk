from functools import partial
from typing import Literal, Union, overload

from FaceEngine import Detection as CoreBodyDetection, FSDKError, Image as CoreImage  # pylint: disable=E0611,E0401

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.detectors.bodydetector import BodyDetection, Landmarks17
from lunavl.sdk.errors.exceptions import LunaSDKException, assertError

from ...errors.errors import LunaVLError
from ..base import BaseEstimator

PreparedBatch = list[tuple[CoreImage, CoreBodyDetection, list[int]]]


def _prepareBatch(detections: list[BodyDetection]) -> PreparedBatch:
    """
    Prepare body detections for core estimator

    Args:
        detections: input body detections

    Returns:
        prepared batch
    """
    res: dict[int, tuple[CoreImage, CoreBodyDetection, list[int]]] = {}
    for idx, detection in enumerate(detections):
        imageId = id(detection.image)
        imageDetections = res.get(imageId)
        coreDetection: CoreBodyDetection = detection.coreEstimation.detection
        if not imageDetections:
            res[imageId] = (detection.image.coreImage, [coreDetection], [idx])
        else:
            imageDetections[1].append(coreDetection)
            imageDetections[2].append(idx)

    return [detectionsPerImage for detectionsPerImage in res.values()]


def _postProcessingBatch(
    error,
    estimatedBatch,
    batchSize: int,
    preparedBatch: PreparedBatch,
) -> Union[list[Landmarks17]]:
    """
    Post-processing core landmarks estimations. Validate error and convert resul to float structure

    Args:
        error: estimation error
        estimatedBatch: core batch estimation
        preparedBatch: prepared batch

    Returns:
        list of landmarks
    """
    assertError(error)
    res: list[Landmarks17] = [None] * batchSize  # type: ignore
    for imageIdx in range(len(preparedBatch)):
        allImageLandmarks = estimatedBatch.getLandmarks(imageIdx)
        detectionIndexes = preparedBatch[imageIdx][2]

        for idx, estimatedLandmarks in enumerate(allImageLandmarks):
            landmarks = Landmarks17(estimatedLandmarks)
            resultIdx = detectionIndexes[idx]
            res[resultIdx] = landmarks
    return res


def _postProcessing(error, estimatedBatch) -> Landmarks17:
    """
    Post-processing one core landmarks estimations. Validate error and convert result to a lunavl structure.

    Args:
        error: core error
        estimatedBatch: estimated core batch

    Returns:
        landmarks
    """
    assertError(error)
    estimatedLandmarks = estimatedBatch[0]
    landmarks = Landmarks17(estimatedLandmarks)
    return landmarks


class BodyLandmarksEstimator(BaseEstimator):
    """
    Body landmarks estimator. Estimator can to predict Landmarks5  or Landmarks68
    """

    @overload  # type: ignore
    def estimate(self, detection: BodyDetection, asyncEstimate: Literal[False] = False) -> Landmarks17: ...

    @overload
    def estimate(self, detection: BodyDetection, asyncEstimate: Literal[True]) -> AsyncTask[Landmarks17]: ...

    def estimate(
        self, detection: BodyDetection, asyncEstimate: bool = False
    ) -> Union[Landmarks17, AsyncTask[Landmarks17]]:
        """
        Estimate landmarks.

        Args:
            detection: body detection
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated states if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        self._validate(detection.image.coreImage, detection.coreEstimation.detection)
        coreDetections = [detection.coreEstimation.detection]
        if asyncEstimate:
            task = self._coreEstimator.asyncDetectLandmarks17(detection.image.coreImage, coreDetections)
            return AsyncTask(task, _postProcessing)
        error, landmarks = self._coreEstimator.detectLandmarks17(detection.image.coreImage, coreDetections)
        return _postProcessing(error, landmarks)

    @overload
    def estimateBatch(
        self, detections: list[BodyDetection], asyncEstimate: Literal[False] = False
    ) -> list[Landmarks17]: ...

    @overload
    def estimateBatch(
        self, detections: list[BodyDetection], asyncEstimate: Literal[True]
    ) -> AsyncTask[list[Landmarks17]]: ...

    def estimateBatch(
        self,
        detections: list[BodyDetection],
        asyncEstimate: bool = False,
    ) -> Union[list[Landmarks17], AsyncTask[list[Landmarks17]]]:
        """
        Estimate landmarks.

        Args:
            detections: body detection
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated states if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        preparedBatch = _prepareBatch(detections)
        coreImages = [image[0] for image in preparedBatch]
        coreDetections = [image[1] for image in preparedBatch]
        batchSize = len(detections)
        self._validate(coreImages, coreDetections)

        if asyncEstimate:
            task = self._coreEstimator.asyncDetectLandmarks17(coreImages, coreDetections)
            return AsyncTask(task, partial(_postProcessingBatch, preparedBatch=preparedBatch, batchSize=batchSize))
        error, estimations = self._coreEstimator.detectLandmarks17(coreImages, coreDetections)
        return _postProcessingBatch(error, estimations, preparedBatch=preparedBatch, batchSize=batchSize)

    def _validate(
        self,
        coreImages: Union[CoreImage, list[CoreImage]],
        detections: Union[CoreBodyDetection, list[list[CoreBodyDetection]]],
    ):
        """
        Validate input data for estimations. Collect errors from single operations and raise complex exception

        Args:
            coreImages: list of core images
            detections: list of list body detection which corresponding to coreImages
        Raises:
            LunaSDKException: if validation are failed or data is not valid
        """
        if isinstance(coreImages, list):
            mainError, imagesErrors = self._coreEstimator.validate(coreImages, detections)
        else:
            mainError, imagesErrors = self._coreEstimator.validate([coreImages], [[detections]])
        if mainError.isOk:
            return
        if mainError.error != FSDKError.ValidationFailed:
            raise LunaSDKException(
                LunaVLError.ValidationFailed.format(mainError.what),
                [LunaVLError.fromSDKError(errors[0]) for errors in imagesErrors],
            )
        if not isinstance(coreImages, list):
            raise LunaSDKException(LunaVLError.fromSDKError(imagesErrors[0][0]))
        errors = []

        for imageErrors in imagesErrors:
            for error in imageErrors:
                if error.isOk:
                    continue
                errors.append(LunaVLError.fromSDKError(error))
                break
            else:
                errors.append(LunaVLError.Ok.format(LunaVLError.Ok.description))
        raise LunaSDKException(
            LunaVLError.BatchedInternalError.format(LunaVLError.fromSDKError(mainError).detail), errors
        )
