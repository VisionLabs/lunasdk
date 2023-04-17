from functools import wraps
from itertools import chain

from lunavl.sdk.estimators.face_estimators.portrait_style import PortraitStyle, PortraitStyleCode
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests import resources
from tests.base import BaseTestClass


class TestPortraitStyle(BaseTestClass):
    """
    Test portrait style estimation.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.estimator = cls.faceEngine.createPortraitStyleEstimator()

    def test_estimate_portrait_style_correctness(self):
        """
        Test portrait style estimator correctness
        """
        estimation = self.estimate(resources.PORTRAIT)
        assert estimation.status == PortraitStyleCode.Portrait

    def estimate(self, image) -> PortraitStyle:
        """Estimate portrait style on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        estimation = self.estimator.estimate(faceDetection)
        assert isinstance(estimation, PortraitStyle)
        return estimation

    def estimateBatch(self, images) -> list[PortraitStyle]:
        """Estimate portrait style on image"""
        imageDetections = self.detector.detect([VLImage.load(filename=name) for name in images])
        estimations = self.estimator.estimateBatch(list(chain(*imageDetections)))
        assert all(isinstance(estimation, PortraitStyle) for estimation in estimations)
        return estimations

    def test_portrait_style_as_dict(self):
        """
        Test method PortraitStyle.asDict
        """
        estimation = self.estimate(resources.FROWNING)
        keys = estimation.asDict().keys()
        assert {"status", "non_portrait", "portrait", "hidden_shoulders"} == keys

    def test_estimate_portrait_style_batch(self):
        """
        Batch portrait style estimation test
        """
        estimations = self.estimateBatch([resources.CLOSED_EYES, resources.PORTRAIT])
        assert estimations[0].status != PortraitStyleCode.Portrait
        assert estimations[1].status == PortraitStyleCode.Portrait
