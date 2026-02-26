import pytest

from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.duty_uniform import DutyUniform, DutyUniformEnum
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import CLEAN_ONE_FACE, DUTY_UNIFORM, FISHEYE

CLEAN_ONE_FACE_IMAGE = VLImage.load(filename=CLEAN_ONE_FACE)
DUTY_UNIFORM_IMAGE = VLImage.load(filename=DUTY_UNIFORM)
FISHEYE_IMAGE = VLImage.load(filename=FISHEYE)


class TestEstimateDutyUniform(BaseTestClass):
    """
    Test estimate  duty uniform
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.estimator = cls.faceEngine.createDutyUniformEstimator()

    def create_data(self, image: VLImage) -> FaceDetection:
        """Helper, create input data for photorealistic face estimator"""
        faceDetection = self.detector.detectOne(image)

        return faceDetection

    def test_estimate_as_dict(self):
        """
        Test 'asDict' method
        """
        detection = self.create_data(DUTY_UNIFORM_IMAGE)
        estimation = self.estimator.estimate(detection)
        dumpedEstimation = estimation.asDict()
        assert dumpedEstimation == {
            "predominant_state": estimation.predominant.value,
            "estimations": {
                "duty_clothes": estimation.dutyClothes,
                "regular_clothes": estimation.regularClothes,
                "unknown": estimation.unknown,
            },
        }

    def test_estimate_batch(self):
        """
        Test estimator batch method
        """
        detections = [self.create_data(img) for img in (DUTY_UNIFORM_IMAGE, CLEAN_ONE_FACE_IMAGE)]
        estimationList = self.estimator.estimateBatch(detections)
        assert isinstance(estimationList, list)
        assert len(estimationList) == len(detections)
        for estimation in estimationList:
            assert isinstance(estimation, DutyUniform)

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
        detection = self.create_data(DUTY_UNIFORM_IMAGE)
        task = self.estimator.estimate(detection, asyncEstimate=True)
        self.assertAsyncEstimation(task, DutyUniform)
        task = self.estimator.estimateBatch([detection] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, DutyUniform)

    def test_face_occlusion_estimation_correctness(self):
        """Test estimation correctness"""
        detection = self.create_data(DUTY_UNIFORM_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert isinstance(estimation, DutyUniform)
        assert estimation.predominant == DutyUniformEnum.DutyClothes
        assert isinstance(estimation.dutyClothes, float)
        assert 1 > estimation.dutyClothes > 0
        assert isinstance(estimation.unknown, float)
        assert 1 > estimation.unknown > 0
        assert isinstance(estimation.regularClothes, float)
        assert 1 > estimation.regularClothes > 0
        detection = self.create_data(FISHEYE_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert estimation.predominant == DutyUniformEnum.Unknown
        detection = self.create_data(CLEAN_ONE_FACE_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert estimation.predominant == DutyUniformEnum.RegularClothes
