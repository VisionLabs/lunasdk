from typing import NamedTuple, Literal, overload, Union, List
from FaceEngine import FaceOcclusionEstimation, FaceOcclusionState, FaceOcclusionType  # pylint: disable=E0611,E0401
from lunavl.sdk.async_task import AsyncTask, DefaultPostprocessingFactory
from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.detectors.facedetector import FaceDetection, Landmarks5
from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarp, FaceWarpedImage


def convertOccludedState(estimation) -> int:
    return int(estimation == FaceOcclusionState.Occluded)


class OcclusionEstimation(NamedTuple):
    """ """

    state: int
    score: float


def getOcclusionState(value: OcclusionEstimation):
    if value[0]:
        occlusion = FaceOcclusionState.Occluded
    else:
        occlusion = FaceOcclusionState.NotOccluded
    return occlusion


class FaceOcclusion(BaseEstimation):

    def __init__(self, coreFaceOcclusion: FaceOcclusionEstimation):
        """
        Init.

        Args:
            coreFaceOcclusion: core occlusion estimation.
        """

        super().__init__(coreFaceOcclusion)

    @property
    def overall(self) -> OcclusionEstimation:
        return OcclusionEstimation(
            convertOccludedState(self._coreEstimation.overallOcclusionState), self._coreEstimation.overallOcclusionScore
        )

    @overall.setter
    def overall(self, overall: OcclusionEstimation):
        occlusion = getOcclusionState(overall)
        self._coreEstimation.overallOcclusionScore = overall[1]
        self._coreEstimation.overallOcclusionState =  occlusion

    @property
    def hairScore(self) -> float:
        return self._coreEstimation.hairOcclusionScore

    @hairScore.setter
    def hairScore(self, value: float):
        self._coreEstimation.hairOcclusionScore = value

    @property
    def forehead(self) -> OcclusionEstimation:
        coreEstimation = self._coreEstimation[FaceOcclusionType.Forehead]
        return OcclusionEstimation(convertOccludedState(coreEstimation[0]), coreEstimation[1])

    @forehead.setter
    def forehead(self, value: OcclusionEstimation):
        occlusion = getOcclusionState(value)
        self._coreEstimation[FaceOcclusionType.Forehead] = (occlusion, value[1])

    @property
    def rightEye(self) -> OcclusionEstimation:
        coreEstimation = self._coreEstimation[FaceOcclusionType.RightEye]
        return OcclusionEstimation(convertOccludedState(coreEstimation[0]), coreEstimation[1])

    @rightEye.setter
    def rightEye(self, value: OcclusionEstimation):
        occlusion = getOcclusionState(value)
        self._coreEstimation[FaceOcclusionType.RightEye] = (occlusion, value[1])

    @property
    def leftEye(self) -> OcclusionEstimation:
        coreEstimation = self._coreEstimation[FaceOcclusionType.LeftEye]
        return OcclusionEstimation(convertOccludedState(coreEstimation[0]), coreEstimation[1])

    @leftEye.setter
    def leftEye(self, value: OcclusionEstimation):
        occlusion = getOcclusionState(value)
        self._coreEstimation[FaceOcclusionType.LeftEye] = (occlusion, value[1])

    @property
    def nose(self) -> OcclusionEstimation:
        coreEstimation = self._coreEstimation[FaceOcclusionType.Nose]
        return OcclusionEstimation(convertOccludedState(coreEstimation[0]), coreEstimation[1])

    @nose.setter
    def nose(self, value: OcclusionEstimation):
        occlusion = getOcclusionState(value)
        self._coreEstimation[FaceOcclusionType.Nose] = (occlusion, value[1])

    @property
    def mouth(self) -> OcclusionEstimation:
        coreEstimation = self._coreEstimation[FaceOcclusionType.Mouth]
        return OcclusionEstimation(convertOccludedState(coreEstimation[0]), coreEstimation[1])

    @mouth.setter
    def mouth(self, value: OcclusionEstimation):
        occlusion = getOcclusionState(value)
        self._coreEstimation[FaceOcclusionType.Mouth] = (occlusion, value[1])

    @property
    def lowerFace(self) -> OcclusionEstimation:
        coreEstimation = self._coreEstimation[FaceOcclusionType.LowerFace]
        return OcclusionEstimation(convertOccludedState(coreEstimation[0]), coreEstimation[1])

    @lowerFace.setter
    def lowerFace(self, value: OcclusionEstimation):
        occlusion = getOcclusionState(value)
        self._coreEstimation[FaceOcclusionType.LowerFace] = (occlusion, value[1])

    def asDict(self):
        overall = self.overall
        lowerFace = self.lowerFace
        mouth = self.mouth
        nose = self.nose
        leftEye = self.leftEye
        rightEye = self.rightEye
        forehead = self.forehead
        return {
            "prediction": overall.state,
            "estimations": {
                "overall_score": overall.score,
                "hair_score": self.hairScore,
                "lower_face_score": lowerFace.score,
                "mouth_score": mouth.score,
                "nose_score": nose.score,
                "left_eye_score": leftEye.score,
                "right_eye_score": rightEye.score,
                "forehead": forehead.score,
            },
            "face_occlusion": {
                "lower_face_occluded": lowerFace.state,
                "mouth_occluded": mouth.state,
                "nose_occluded": nose.state,
                "left_eye_occluded": leftEye.state,
                "right_eye_occluded": rightEye.state,
                "forehead_occluded": forehead.state,
            },
        }


class WarpWithLandmarks(NamedTuple):
    """
    Structure for transferring a detector landmarks estimation and its warp.
    Attributes
        warp (Union[FaceWarp, FaceWarpedImage]): warp core image
        landmarks (Union[Landmarks5, Landmarks68]): landmarks estimation
    """

    warp: Union[FaceWarp, FaceWarpedImage]
    landmarks: Landmarks5


POST_PROCESSING_FACE_OCCLUSION = DefaultPostprocessingFactory(FaceOcclusion)


class FaceOcclusionEstimator(BaseEstimator):
    """
    Face occlusion estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self, warpWithLandmarks: WarpWithLandmarks, asyncEstimate: Literal[False] = False
    ) -> FaceOcclusion: ...

    @overload
    def estimate(
        self, warpWithLandmarks: WarpWithLandmarks, asyncEstimate: Literal[True]
    ) -> AsyncTask[FaceOcclusion]: ...

    def estimate(  # type: ignore
        self,
        warpWithLandmarks: WarpWithLandmarks,
        asyncEstimate: bool = False,
    ) -> Union[FaceOcclusion, AsyncTask[FaceOcclusion]]:
        """
        Estimate mouth state on warp.

        Args:
            warpWithLandmarks: core warp with transformed landmarks
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated states if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """

        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(
                warpWithLandmarks[0].warpedImage.coreImage, warpWithLandmarks[1].coreEstimation
            )
            return AsyncTask(task, POST_PROCESSING_FACE_OCCLUSION.postProcessing)
        error, estimation = self._coreEstimator.estimate(
            warpWithLandmarks[0].warpedImage.coreImage, warpWithLandmarks[1].coreEstimation
        )
        return POST_PROCESSING_FACE_OCCLUSION.postProcessing(error, estimation)

    # pylint: disable=W0221
    def estimateBatch(
        self, warpWithLandmarksList: List[WarpWithLandmarks], asyncEstimate: bool = False
    ) -> Union[List[FaceOcclusion], AsyncTask[List[FaceOcclusion]]]:
        """
        Batch estimate face occlusion state on warps.

        Args:
            warpWithLandmarksList: list of core warp with transformed landmarks
            asyncEstimate: estimate or run estimation in background

        Returns:
            list of estimated states if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if warps count not equals landmarks count
        """
        coreImages = [row[0].warpedImage.coreImage for row in warpWithLandmarksList]
        landmarks = [row[1].coreEstimation for row in warpWithLandmarksList]
        validateInputByBatchEstimator(self._coreEstimator, coreImages, landmarks)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages, landmarks)
            return AsyncTask(task, POST_PROCESSING_FACE_OCCLUSION.postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(coreImages, landmarks)
        return POST_PROCESSING_FACE_OCCLUSION.postProcessingBatch(error, estimations)
