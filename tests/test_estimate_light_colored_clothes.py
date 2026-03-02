import pytest

from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.light_colored_clothes import (
    LightColoredClothes,
    LightColoredClothesEnum,
)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import CLEAN_ONE_FACE, CROP_DISGUST, ROTATED0

CLEAN_ONE_FACE_IMAGE = VLImage.load(filename=CLEAN_ONE_FACE)
ROTATED0_IMAGE = VLImage.load(filename=ROTATED0)
WARP_FACE_WITH_EYEGLASSES_IMAGE = VLImage.load(filename=CROP_DISGUST)


class TestEstimateLightColoredClothes(BaseTestClass):
    """
    Test estimate light colored clothes
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.estimator = cls.faceEngine.createLightColoredClothesEstimator()

    def create_data(self, image: VLImage) -> FaceDetection:
        """Helper, create input data for photorealistic face estimator"""
        faceDetection = self.detector.detectOne(image)

        return faceDetection

    def test_estimate_as_dict(self):
        """
        Test 'asDict' method
        """
        detection = self.create_data(CLEAN_ONE_FACE_IMAGE)
        estimation = self.estimator.estimate(detection)
        dumpedEstimation = estimation.asDict()
        assert dumpedEstimation == {
            "predominant": estimation.predominant.value,
            "estimations": {
                "light_clothes": estimation.lightClothes,
                "dark_clothes": estimation.darkClothes,
                "unknown": estimation.unknown,
            },
        }

    def test_estimate_batch(self):
        """
        Test estimator batch method
        """
        detections = [self.create_data(img) for img in (ROTATED0_IMAGE, CLEAN_ONE_FACE_IMAGE)]
        estimationList = self.estimator.estimateBatch(detections)
        assert isinstance(estimationList, list)
        assert len(estimationList) == len(detections)
        for estimation in estimationList:
            assert isinstance(estimation, LightColoredClothes)

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
        self.assertAsyncEstimation(task, LightColoredClothes)
        task = self.estimator.estimateBatch([detection] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, LightColoredClothes)

    def test_face_occlusion_estimation_correctness(self):
        """Test estimation correctness"""
        detection = self.create_data(CLEAN_ONE_FACE_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert isinstance(estimation, LightColoredClothes)
        assert estimation.predominant == LightColoredClothesEnum.DarkClothes
        assert isinstance(estimation.lightClothes, float)
        assert 1 > estimation.lightClothes > 0
        assert isinstance(estimation.unknown, float)
        assert 1 > estimation.unknown > 0
        assert isinstance(estimation.darkClothes, float)
        assert 1 > estimation.darkClothes > 0
        detection = self.create_data(WARP_FACE_WITH_EYEGLASSES_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert estimation.predominant == LightColoredClothesEnum.Unknown
        detection = self.create_data(ROTATED0_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert estimation.predominant == LightColoredClothesEnum.LightClothes
