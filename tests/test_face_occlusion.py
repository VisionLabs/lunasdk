import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.face_occlusion import FaceOcclusion, OcclusionEstimation, WarpWithLandmarks
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType, FaceEngineSettingsProvider
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import CLEAN_ONE_FACE, MASK_FULL

CLEAN_ONE_FACE_IMAGE = VLImage.load(filename=CLEAN_ONE_FACE)
MASK_FULL_IMAGE = VLImage.load(filename=MASK_FULL)


class TestEstimateFaceOcclusion(BaseTestClass):
    """
    Test estimate face occlusion.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.faceOcclusionEstimator = cls.faceEngine.createFaceOcclusionEstimator()

    def create_data(self, image: VLImage) -> WarpWithLandmarks:
        """Helper, create input data for face occlusion estimator"""
        faceDetection = self.detector.detectOne(image)
        warp = self.warper.warp(faceDetection)
        landMarks5Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")
        warpWithLandmarks = WarpWithLandmarks(warp, landMarks5Transformation)
        return warpWithLandmarks

    def assert_occlusion_reply(self, occlusionDict: dict) -> None:
        """
        Assert occlusion dict
        Args:
            occlusionDict: dict with eyes detection reply
        """
        assert occlusionDict.keys() == {"prediction", "estimations", "face_occlusion"}
        assert isinstance(occlusionDict["prediction"], int)
        assert 0 <= occlusionDict["prediction"] <= 1
        assert occlusionDict["estimations"].keys() == {
            "overall_score",
            "hair_score",
            "lower_face_score",
            "mouth_score",
            "nose_score",
            "left_eye_score",
            "right_eye_score",
            "forehead_score",
        }
        assert occlusionDict["face_occlusion"].keys() == {
            "lower_face_occluded",
            "mouth_occluded",
            "nose_occluded",
            "left_eye_occluded",
            "right_eye_occluded",
            "forehead_occluded",
        }

        for field, value in occlusionDict["estimations"].items():
            assert isinstance(value, float), field
            assert 0 <= value <= 1, field

        for field, value in occlusionDict["face_occlusion"].items():
            assert isinstance(value, int), field
            assert 0 <= value <= 1, field

    def test_estimate_occlusion_as_dict(self):
        """
        Test occlusion estimator 'asDict' method
        """
        warpWithLandmarks = self.create_data(MASK_FULL_IMAGE)
        occlusionDict = self.faceOcclusionEstimator.estimate(warpWithLandmarks).asDict()
        self.assert_occlusion_reply(occlusionDict)

    def test_estimate_batch(self):
        """
        Test face occlusion estimator with two faces
        """
        warpWithLandmarksList = [self.create_data(img) for img in (MASK_FULL_IMAGE, CLEAN_ONE_FACE_IMAGE)]
        occlusionList = self.faceOcclusionEstimator.estimateBatch(warpWithLandmarksList)
        assert isinstance(occlusionList, list)
        assert len(occlusionList) == len(warpWithLandmarksList)
        for occlusion in occlusionList:
            assert isinstance(occlusion, FaceOcclusion)

    def test_estimate_batch_invalid_input(self):
        """
        Test batch face occlusion estimator with invalid input
        """
        with pytest.raises(LunaSDKException) as e:
            self.faceOcclusionEstimator.estimateBatch([], [])
        assert e.value.error.errorCode == LunaVLError.InvalidSpanSize.errorCode

    def test_async_estimate_face_occlusion(self):
        """
        Test async estimate face occlusion
        """
        warpWithLandmarks = self.create_data(MASK_FULL_IMAGE)
        task = self.faceOcclusionEstimator.estimate(warpWithLandmarks, asyncEstimate=True)
        self.assertAsyncEstimation(task, FaceOcclusion)
        task = self.faceOcclusionEstimator.estimateBatch([warpWithLandmarks] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, FaceOcclusion)

    def test_some_face_occlusion_settings(self):
        """Face occlusion settings test."""
        settings = FaceEngineSettingsProvider()
        settings.faceOcclusionEstimatorSettings.overallOcclusionThreshold = 0.001
        faceEngine = VLFaceEngine(faceEngineConf=settings)
        faceOcclusionEstimator = faceEngine.createFaceOcclusionEstimator()
        warpWithLandmarks = self.create_data(CLEAN_ONE_FACE_IMAGE)
        estimation = faceOcclusionEstimator.estimate(warpWithLandmarks)
        assert estimation.overall.state == 0
        assert estimation.overall.score == 0
        settings.faceOcclusionEstimatorSettings.overallOcclusionThreshold = 0.1
        settings.faceOcclusionEstimatorSettings.normalHairCoeff = 0.001
        faceEngine = VLFaceEngine(faceEngineConf=settings)
        faceOcclusionEstimator = faceEngine.createFaceOcclusionEstimator()
        estimation = faceOcclusionEstimator.estimate(warpWithLandmarks)
        assert estimation.overall.state == 1
        assert estimation.overall.score > 0.1
        assert estimation.hairScore > 0.1

    def test_face_occlusion_estimation_correctness(self):
        """Test mutable field of FaceOcclusion structure"""
        warpWithLandmarks = self.create_data(MASK_FULL_IMAGE)
        estimation = self.faceOcclusionEstimator.estimate(warpWithLandmarks)
        estimation.overall = OcclusionEstimation(0, 0.1)
        estimation.forehead = OcclusionEstimation(0, 0.2)
        estimation.rightEye = OcclusionEstimation(1, 0.3)
        estimation.leftEye = OcclusionEstimation(1, 0.4)
        estimation.nose = OcclusionEstimation(0, 0.5)
        estimation.mouth = OcclusionEstimation(0, 0.6)
        estimation.lowerFace = OcclusionEstimation(0, 0.7)
        estimation.hair = 0.8

        assert estimation.overall.state == 0
        assert 0.1 == pytest.approx(estimation.overall.score, 0.001)

        assert estimation.forehead.state == 0
        assert 0.2 == pytest.approx(estimation.forehead.score, 0.001)

        assert estimation.rightEye.state == 1
        assert 0.3 == pytest.approx(estimation.rightEye.score, 0.001)

        assert estimation.leftEye.state == 1
        assert 0.4 == pytest.approx(estimation.leftEye.score, 0.001)

        assert estimation.nose.state == 0
        assert 0.5 == pytest.approx(estimation.nose.score, 0.001)

        assert estimation.mouth.state == 0
        assert 0.6 == pytest.approx(estimation.mouth.score, 0.001)

        assert estimation.lowerFace.state == 0
        assert 0.7 == pytest.approx(estimation.lowerFace.score, 0.001)

        assert 0.8 == pytest.approx(estimation.hair, 0.001)
