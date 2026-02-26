import pytest

from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.photorealistic_face import PhotorealisticPrediction
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType, FaceEngineSettingsProvider
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import CLEAN_ONE_FACE, RAFAEL

CLEAN_ONE_FACE_IMAGE = VLImage.load(filename=CLEAN_ONE_FACE)
RAFAEL_IMAGE = VLImage.load(filename=RAFAEL)


class TestEstimatePhotorealisticFacePrediction(BaseTestClass):
    """
    Test estimate photorealistic face prediction
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.estimator = cls.faceEngine.createPhotorealisticFaceEstimator()

    def create_data(self, image: VLImage) -> FaceDetection:
        """Helper, create input data for photorealistic face estimator"""
        faceDetection = self.detector.detectOne(image)

        return faceDetection

    def test_estimate_occlusion_as_dict(self):
        """
        Test 'asDict' method
        """
        detection = self.create_data(CLEAN_ONE_FACE_IMAGE)
        estimation = self.estimator.estimate(detection)
        dumpedEstimation = self.estimator.estimate(detection).asDict()
        assert dumpedEstimation == {"score": estimation.score, "status": estimation.status}

    def test_estimate_batch(self):
        """
        Test estimator batch method
        """
        detections = [self.create_data(img) for img in (RAFAEL_IMAGE, CLEAN_ONE_FACE_IMAGE)]
        estimationList = self.estimator.estimateBatch(detections)
        assert isinstance(estimationList, list)
        assert len(estimationList) == len(detections)
        for estimation in estimationList:
            assert isinstance(estimation, PhotorealisticPrediction)

    def test_estimate_batch_invalid_input(self):
        """
        Test batch  method with invalid input
        """
        with pytest.raises(LunaSDKException) as e:
            self.estimator.estimateBatch([], [])
        assert e.value.error.errorCode == LunaVLError.InvalidSpanSize.errorCode

    def test_async_estimate(self):
        """
        Test async estimate
        """
        detection = self.create_data(CLEAN_ONE_FACE_IMAGE)
        task = self.estimator.estimate(detection, asyncEstimate=True)
        self.assertAsyncEstimation(task, PhotorealisticPrediction)
        task = self.estimator.estimateBatch([detection] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, PhotorealisticPrediction)

    def test_estimator_settings_settings(self):
        """Estimator settings test."""
        settings = FaceEngineSettingsProvider()
        settings.photorealisticFaceEstimator.threshold = 0.001
        faceEngine = VLFaceEngine(faceEngineConf=settings)
        estimator = faceEngine.createPhotorealisticFaceEstimator()
        detection = self.create_data(CLEAN_ONE_FACE_IMAGE)
        estimation = estimator.estimate(detection)
        assert estimation.status
        settings.photorealisticFaceEstimator.threshold = 0.9999
        faceEngine = VLFaceEngine(faceEngineConf=settings)
        estimator = faceEngine.createPhotorealisticFaceEstimator()
        estimation = estimator.estimate(detection)
        assert not estimation.status

    def test_face_occlusion_estimation_correctness(self):
        """Test estimation correctness"""
        detection = self.create_data(CLEAN_ONE_FACE_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert isinstance(estimation, PhotorealisticPrediction)
        assert estimation.status is True
        assert isinstance(estimation.score, float)
        assert 1 > estimation.score > 0
        detection = self.create_data(RAFAEL_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert estimation.status is False
