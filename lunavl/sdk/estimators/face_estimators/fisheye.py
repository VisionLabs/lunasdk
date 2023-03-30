"""
Module contains a fisheye estimator.

See `fisheye`_.
"""
import warnings
from typing import Dict, List, Union

from FaceEngine import FishEye as CoreFishEye, FishEyeEstimation as CoreFishEyeEstimation  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.detectors.facedetector import FaceDetection

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ..base import BaseEstimator, ImageWithFaceDetection
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage


class Fisheye(BaseEstimation):
    """
    Fisheye. Estimation of fisheye effect on face detection (https://en.wikipedia.org/wiki/Fisheye_lens).

    Estimation properties:

        - status
        - score
    """

    #  pylint: disable=W0235
    def __init__(self, coreFishEye: CoreFishEyeEstimation):
        """
        Init.

        Args:
            coreFishEye: core fisheye estimation.
        """

        super().__init__(coreFishEye)

    @property
    def score(self) -> float:
        """Prediction score"""
        return self._coreEstimation.score

    @property
    def status(self) -> bool:
        """
        Prediction status.
        Returns:
            True if image contains the fisheye effect otherwise false
        """
        return True if self._coreEstimation.result == CoreFishEye.FishEyeEffect else False

    def asDict(self) -> Dict[str, Union[float, bool]]:
        """Convert estimation to dict."""
        return {"score": self.score, "status": self.status}


POST_PROCESSING = DefaultPostprocessingFactory(Fisheye)


class FisheyeEstimator(BaseEstimator):
    """
    Fisheye effect estimator. Work on face detections
    """

    def estimate(  # type: ignore
        self,
        image: Union[FaceDetection, ImageWithFaceDetection, FaceWarp, FaceWarpedImage],
        asyncEstimate: bool = False,
    ) -> Union[Fisheye, AsyncTask[Fisheye]]:
        """
        Estimate fisheye.

        Args:
            image: image with face detection (deprecated), or warp
            asyncEstimate: estimate or run estimation in background
        Returns:
            fisheye estimation if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """

        if isinstance(image, (FaceWarp, FaceWarpedImage)):
            if not asyncEstimate:
                error, result = self._coreEstimator.estimate_warp(image.warpedImage.coreImage)
                return POST_PROCESSING.postProcessing(error, result)
            task = self._coreEstimator.asyncEstimate_warp(image.warpedImage.coreImage)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        elif isinstance(image, (FaceDetection, ImageWithFaceDetection)):
            warnings.warn("fisheye crop is deprecated", DeprecationWarning, stacklevel=2)
            if not asyncEstimate:
                error, result = self._coreEstimator.estimate(image.image.coreImage, image.boundingBox.coreEstimation)
                return POST_PROCESSING.postProcessing(error, result)
            task = self._coreEstimator.asyncEstimate(image.image.coreImage, image.boundingBox.coreEstimation)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        else:
            raise RuntimeError(f"Unsupported image type: {image}")

    def estimateBatch(
        self,
        batch: Union[List[ImageWithFaceDetection], List[FaceDetection], List[FaceWarp], List[FaceWarpedImage]],
        asyncEstimate: bool = False,
    ) -> Union[List[Fisheye], AsyncTask[List[Fisheye]]]:
        """
        Estimate fisheye batch.

        Args:
            batch: list of image with face detection or face detections (deprecated), or warps
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of fisheye estimations if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """

        if isinstance(batch[0], (FaceWarp, FaceWarpedImage)):
            coreImages = [row.warpedImage.coreImage for row in batch]  # type: ignore
            validateInputByBatchEstimator(self._coreEstimator, coreImages)
            if not asyncEstimate:
                error, result = self._coreEstimator.estimate_warp(coreImages)
                return POST_PROCESSING.postProcessingBatch(error, result)
            task = self._coreEstimator.asyncEstimate_warp(coreImages)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        elif isinstance(batch[0], (ImageWithFaceDetection, FaceDetection)):
            warnings.warn("fisheye crop is deprecated", DeprecationWarning, stacklevel=2)
            coreImages = [row.image.coreImage for row in batch]  # type: ignore
            detections = [row.boundingBox.coreEstimation for row in batch]  # type: ignore
            validateInputByBatchEstimator(self._coreEstimator, coreImages, detections)
            if not asyncEstimate:
                error, result = self._coreEstimator.estimate(coreImages, detections)
                return POST_PROCESSING.postProcessingBatch(error, result)
            task = self._coreEstimator.asyncEstimate(coreImages, detections)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        else:
            raise RuntimeError(f"Unsupported image type: {batch[0]}")
