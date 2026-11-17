from functools import partial
from typing import Literal, overload

from FaceEngine import Detection as CoreFaceDetection, FSDKError, Image as CoreImage  # pylint: disable=E0611,E0401

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.detectors.facedetector import FaceDetection, FaceLandmarks, Landmarks5, Landmarks68
from lunavl.sdk.errors.exceptions import LunaSDKException, assertError

from ...errors.errors import LunaVLError
from ..base import BaseEstimator

PreparedBatch = list[tuple[CoreImage, list[CoreFaceDetection], list[int]]]


def _prepareBatch(detections: list[FaceDetection]) -> PreparedBatch:
    """
    Prepare face detections for core estimator

    Args:
        detections: input face detections

    Returns:
        List of Tuples. Each tuple is image, Core detections on the image, detection index in original detection array.
    """
    res: dict[int, tuple[CoreImage, list[CoreFaceDetection], list[int]]] = {}
    for idx, detection in enumerate(detections):
        coreImage = detection.image.coreImage
        coreDetection: CoreFaceDetection = detection.coreEstimation.detection
        imageKey = id(coreImage)
        imageDetections = res.get(imageKey)
        if imageDetections is None:
            res[imageKey] = (coreImage, [coreDetection], [idx])
        else:
            imageDetections[1].append(coreDetection)
            imageDetections[2].append(idx)

    return [detectionsPerImage for detectionsPerImage in res.values()]


@overload
def _postProcessingBatch(
    error,
    estimatedBatch,
    batchSize: int,
    preparedBatch: PreparedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks5],
) -> list[Landmarks5 | None]: ...


@overload
def _postProcessingBatch(
    error,
    estimatedBatch,
    batchSize: int,
    preparedBatch: PreparedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks68],
) -> list[Landmarks68 | None]: ...


def _postProcessingBatch(
    error,
    estimatedBatch,
    batchSize: int,
    preparedBatch: PreparedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68],
) -> list[Landmarks68 | None] | list[Landmarks5 | None]:
    """
    Post processing core landmarks estimations. Validate error and convert resul to float structure

    Args:
        error: estimation error
        estimatedBatch: core batch estimation
        preparedBatch: prepared batch
        landmarksType: estimated landmarks type

    Returns:
        list of landmarks, an element is None if the core did not return landmarks for its detection

    Raises:
        LunaSDKException: if the core returned Ok without landmarks for all the requested detections,
            or returned some but not all landmarks of one image
    """
    assertError(error)
    if landmarksType == FaceLandmarks.Landmarks5:
        landmarksClass = Landmarks5
        getter = estimatedBatch.getLandmarks5
        res: list[Landmarks5 | None] = [None for _ in range(batchSize)]
    else:
        landmarksClass = Landmarks68  # type: ignore
        getter = estimatedBatch.getLandmarks68
        res = [None for _ in range(batchSize)]
    for imageIdx in range(len(preparedBatch)):
        allImageLandmarks = getter(imageIdx)
        detectionIndexes = preparedBatch[imageIdx][2]
        if allImageLandmarks and len(allImageLandmarks) != len(detectionIndexes):
            raise LunaSDKException(
                LunaVLError.BatchedInternalError.format(
                    f"core landmarks estimation returned {len(allImageLandmarks)} results "
                    f"for {len(detectionIndexes)} requested detections"
                )
            )

        for idx, estimatedLandmarks in enumerate(allImageLandmarks):
            landmarks = landmarksClass(estimatedLandmarks)
            resultIdx = detectionIndexes[idx]
            res[resultIdx] = landmarks
    if batchSize and all(landmarks is None for landmarks in res):
        raise LunaSDKException(
            LunaVLError.LandmarksNotEstimated.format(
                f"core landmarks estimation answered Ok without landmarks for {batchSize} requested detections"
            )
        )
    return res


@overload
def _postProcessing(
    error,
    estimatedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks5],
) -> Landmarks5: ...


@overload
def _postProcessing(
    error,
    estimatedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks68],
) -> Landmarks68: ...


def _postProcessing(
    error, estimatedBatch, landmarksType: Literal[FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68]
) -> Landmarks5 | Landmarks68:
    """
    Post processing one core landmarks estimations. Validate error and convert result to a lunavl structure.

    Args:
        error: core error
        estimatedBatch: estimated core batch
        landmarksType: estimated landmarks type

    Returns:
        Landmarks5 or Landmarks68

    Raises:
        LunaSDKException: if the core answered Ok without landmarks for the requested detection
    """
    assertError(error)
    landmarksClass = Landmarks5 if landmarksType == FaceLandmarks.Landmarks5 else Landmarks68
    if not estimatedBatch:
        raise LunaSDKException(
            LunaVLError.LandmarksNotEstimated.format(
                "core landmarks estimation answered Ok without landmarks for the requested detection"
            )
        )
    estimatedLandmarks = estimatedBatch[0]
    landmarks = landmarksClass(estimatedLandmarks)
    return landmarks  # type: ignore


class FaceLandmarksEstimator(BaseEstimator):
    """
    Face landmarks estimator. Estimator can to predict Landmarks5  or Landmarks68
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks5],
        asyncEstimate: Literal[False] = False,
    ) -> Landmarks5: ...

    @overload
    def estimate(
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks5],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[Landmarks5]: ...

    @overload
    def estimate(  # type: ignore
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks68],
        asyncEstimate: Literal[False],
    ) -> Landmarks68: ...

    @overload
    def estimate(
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks68],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[Landmarks68]: ...

    def estimate(
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68],
        asyncEstimate: bool = False,
    ):
        """
        Estimate landmarks of a face detection.

        Args:
            detection: face detection
            landmarksType: landmarks type for estimation
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated landmarks, an async task if asyncEstimate is True
        Raises:
            LunaSDKException: if estimation failed or the core answered Ok without landmarks
        """
        if landmarksType == FaceLandmarks.Landmarks5:
            postProcessing = partial(_postProcessing, landmarksType=FaceLandmarks.Landmarks5)
            estimator = (
                self._coreEstimator.asyncDetectLandmarks5 if asyncEstimate else self._coreEstimator.detectLandmarks5
            )
        else:
            postProcessing = partial(_postProcessing, landmarksType=FaceLandmarks.Landmarks68)
            estimator = (
                self._coreEstimator.asyncDetectLandmarks68 if asyncEstimate else self._coreEstimator.detectLandmarks68
            )
        self._validate(detection.image.coreImage, detection.coreEstimation.detection)
        if asyncEstimate:
            task = estimator(detection.image.coreImage, [detection.coreEstimation.detection])
            return AsyncTask(task, postProcessing)
        error, landmarks = estimator(detection.image.coreImage, [detection.coreEstimation.detection])
        return postProcessing(error, landmarks)

    @overload
    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks5],
        asyncEstimate: Literal[False] = False,
    ) -> list[Landmarks5 | None]: ...

    @overload
    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks5],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[list[Landmarks5 | None]]: ...

    @overload
    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks68],
        asyncEstimate: Literal[False],
    ) -> list[Landmarks68 | None]: ...

    @overload
    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks68],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[list[Landmarks68 | None]]: ...

    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68],
        asyncEstimate: bool = False,
    ) -> (
        list[Landmarks5 | None]
        | list[Landmarks68 | None]
        | AsyncTask[list[Landmarks68 | None]]
        | AsyncTask[list[Landmarks5 | None]]
    ):
        """
        Estimate landmarks of face detections.

        Args:
            detections: face detections
            landmarksType: landmarks type for estimation
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated landmarks per detection, an element is None if the core did not return landmarks for it,
            an async task if asyncEstimate is True
        Raises:
            LunaSDKException: if estimation failed or the core answered Ok without landmarks for all detections
        """
        preparedBatch = _prepareBatch(detections)
        coreImages = [image[0] for image in preparedBatch]
        coreDetections = [image[1] for image in preparedBatch]
        batchSize = len(detections)
        self._validate(coreImages, coreDetections)
        if landmarksType == FaceLandmarks.Landmarks5:
            postProcessing = partial(
                _postProcessingBatch,
                preparedBatch=preparedBatch,
                batchSize=batchSize,
                landmarksType=FaceLandmarks.Landmarks5,
            )
            estimator = (
                self._coreEstimator.asyncDetectLandmarks5 if asyncEstimate else self._coreEstimator.detectLandmarks5
            )
        else:
            postProcessing = partial(
                _postProcessingBatch,
                preparedBatch=preparedBatch,
                batchSize=batchSize,
                landmarksType=FaceLandmarks.Landmarks68,
            )
            estimator = (
                self._coreEstimator.asyncDetectLandmarks68 if asyncEstimate else self._coreEstimator.detectLandmarks68
            )

        if asyncEstimate:
            task = estimator(coreImages, coreDetections)
            return AsyncTask(task, postProcessing)
        error, estimations = estimator(coreImages, coreDetections)
        return postProcessing(error, estimations, preparedBatch=preparedBatch, batchSize=batchSize)

    def _validate(
        self,
        coreImages: CoreImage | list[CoreImage],
        detections: CoreFaceDetection | list[list[CoreFaceDetection]],
    ):
        """
        Validate input data for estimations. Collect errors from single operations and raise complex exception

        Args:
            coreImages: list of core images
            detections: list of list face detection which corresponding to coreImages
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
