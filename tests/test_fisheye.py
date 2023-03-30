from itertools import chain

import pytest

from lunavl.sdk.estimators.face_estimators.fisheye import Fisheye
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import FISHEYE, FROWNING


class TestFisheyeEffect(BaseTestClass):
    """
    Test fisheye estimation.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.fisheyeEstimator = cls.faceEngine.createFisheyeEstimator()

    def test_estimate_fisheye_correctness(self):
        """
        Test fisheye estimator correctness
        """
        estimation = self.estimate(FISHEYE)
        assert estimation.status
        assert 0 <= estimation.score <= 1

    def estimate(self, image: str = FROWNING, warped: bool = True) -> Fisheye:
        """Estimate fisheye on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        if warped:
            warp = self.warper.warp(faceDetection)
            estimation = self.fisheyeEstimator.estimate(warp.warpedImage)
        else:
            estimation = self.fisheyeEstimator.estimate(faceDetection)
        assert isinstance(estimation, Fisheye)
        return estimation

    def estimateBatch(self, images, warped) -> list[Fisheye]:
        """Estimate fisheye on image"""
        imageDetections = self.detector.detect([VLImage.load(filename=name) for name in images])
        if warped:
            warps = [self.warper.warp(res[0]) for res in imageDetections]
            estimations = self.fisheyeEstimator.estimateBatch(warps)
        else:
            estimations = self.fisheyeEstimator.estimateBatch(list(chain(*imageDetections)))
        assert all(isinstance(estimation, Fisheye) for estimation in estimations)
        return estimations

    @pytest.mark.parametrize("warped", [True, False])
    def test_estimate_fisheye(self, warped):
        """
        Simple fisheye estimation
        """
        estimation = self.estimate(FROWNING, warped=warped)
        assert not estimation.status
        assert 0 <= estimation.score <= 1

    @pytest.mark.parametrize("warped", [True, False])
    def test_fisheye_as_dict(self, warped):
        """
        Test method Fisheye.asDict
        """
        estimation = self.estimate(FROWNING, warped=warped)
        assert {
            "status": estimation.status,
            "score": estimation.score,
        } == estimation.asDict()

    @pytest.mark.parametrize("warped", [True, False])
    def test_estimate_fisheye_batch(self, warped):
        """
        Batch fisheye estimation test
        """
        estimations = self.estimateBatch([FROWNING, FISHEYE], warped=warped)
        assert not estimations[0].status
        assert estimations[1].status
