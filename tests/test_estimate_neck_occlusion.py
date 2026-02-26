import pytest

from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.neck_occlusion import (
    NeckOcclusion,
    NeckOcclusionEnum,
)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import CLEAN_ONE_FACE, FISHEYE, PORTRAIT

CLEAN_ONE_FACE_IMAGE = VLImage.load(filename=CLEAN_ONE_FACE)
PORTRAIT_IMAGE = VLImage.load(filename=PORTRAIT)
FISHEYE_IMAGE = VLImage.load(filename=FISHEYE)


class TestEstimateNeckOcclusion(BaseTestClass):
    """
    Test estimate neck occlusion status
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.estimator = cls.faceEngine.createNeckOcclusionEstimator()

    def create_data(self, image: VLImage) -> FaceDetection:
        """Helper, create input data for photorealistic face estimator"""
        faceDetection = self.detector.detectOne(image)

        return faceDetection

    def test_estimate_as_dict(self):
        """
        Test 'asDict' method
        """
        detection = self.create_data(PORTRAIT_IMAGE)
        estimation = self.estimator.estimate(detection)
        dumpedEstimation = estimation.asDict()
        assert dumpedEstimation == {
            "predominant_state": estimation.predominant.value,
            "estimations": {
                "open_neck": estimation.openNeck,
                "occluded_neck": estimation.occludedNeck,
                "unknown": estimation.unknown,
            },
        }

    def test_estimate_batch(self):
        """
        Test estimator batch method
        """
        detections = [self.create_data(img) for img in (PORTRAIT_IMAGE, CLEAN_ONE_FACE_IMAGE)]
        estimationList = self.estimator.estimateBatch(detections)
        assert isinstance(estimationList, list)
        assert len(estimationList) == len(detections)
        for estimation in estimationList:
            assert isinstance(estimation, NeckOcclusion)

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
        detection = self.create_data(PORTRAIT_IMAGE)
        task = self.estimator.estimate(detection, asyncEstimate=True)
        self.assertAsyncEstimation(task, NeckOcclusion)
        task = self.estimator.estimateBatch([detection] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, NeckOcclusion)

    def test_face_occlusion_estimation_correctness(self):
        """Test estimation correctness"""
        detection = self.create_data(CLEAN_ONE_FACE_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert isinstance(estimation, NeckOcclusion)
        assert estimation.predominant == NeckOcclusionEnum.OpenNeck
        assert isinstance(estimation.openNeck, float)
        assert 1 > estimation.openNeck > 0
        assert isinstance(estimation.unknown, float)
        assert 1 > estimation.unknown > 0
        assert isinstance(estimation.occludedNeck, float)
        assert 1 > estimation.occludedNeck > 0
        detection = self.create_data(FISHEYE_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert estimation.predominant == NeckOcclusionEnum.Unknown
        detection = self.create_data(PORTRAIT_IMAGE)
        estimation = self.estimator.estimate(detection)
        assert estimation.predominant == NeckOcclusionEnum.OccludedNeck
