import unittest
from typing import Dict

from lunavl.sdk.estimators.face_estimators.facial_hair import FacialHair, FacialHairEstimator
from lunavl.sdk.faceengine.setting_provider import DetectorType
from tests.base import BaseTestClass
from tests.resources import WARP_FACE_WITH_BEARD, WARP_FACE_WITH_MUSTACHE, WARP_FACE_WITH_STUBBLE, WARP_NO_FACIAL_HAIR


@unittest.skip("No plan in sources")
class TestFacialHair(BaseTestClass):
    """
    Test estimate glasses.
    """

    facialHairEstimator: FacialHairEstimator

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.facialHairEstimator = cls.faceEngine.createFacialHairEstimator()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createFaceWarper()

    def assertEstimation(self, facialHair: FacialHair, expectedEstimationResults: Dict[str, float]):
        """
        Function checks if the instance belongs to the FacialHair class and compares the result with what is expected.

        Args:
            facialHair: facial hair estimation object
            expectedEstimationResults: dictionary with result
        """
        assert isinstance(facialHair, FacialHair), f"{facialHair.__class__} is not {FacialHair}"
        self.assertEqual(facialHair.asDict(), expectedEstimationResults)

    def test_estimate_no_hair(self):
        """
        Test facial hair estimations without facial hair on the face
        """
        expectedResult = expectedResult = {"beard": 0.0, "mustache": 0.0, "noHair": 0.0, "stubble": 0.0}
        faceDetection = self.detector.detectOne(WARP_NO_FACIAL_HAIR)
        warp = self.warper.warp(faceDetection)
        facialHair = self.facialHairEstimator.estimate(warp)
        self.assertEstimation(facialHair, expectedResult)

    def test_estimate_beard(self):
        """
        Test facial hair estimation with beard on the face
        """
        expectedResult = {"beard": 0.0, "mustache": 0.0, "noHair": 0.0, "stubble": 0.0}
        faceDetection = self.detector.detectOne(WARP_FACE_WITH_BEARD)
        warp = self.warper.warp(faceDetection)
        facialHair = self.facialHairEstimator.estimate(warp)
        self.assertEstimation(facialHair, expectedResult)

    def test_estimate_stubble(self):
        """
        Test facial hair estimations with stubble on the face
        """
        expectedResult = {"beard": 0.0, "mustache": 0.0, "noHair": 0.0, "stubble": 0.0}
        faceDetection = self.detector.detectOne(WARP_FACE_WITH_STUBBLE)
        warp = self.warper.warp(faceDetection)
        facialHair = self.facialHairEstimator.estimate(warp)
        self.assertEstimation(facialHair, expectedResult)

    def test_estimate_mustache(self):
        """
        Test facial hair estimations with mustache on the face
        """
        expectedResult = {"beard": 0.0, "mustache": 0.0, "noHair": 0.0, "stubble": 0.0}
        faceDetection = self.detector.detectOne(WARP_FACE_WITH_MUSTACHE)
        warp = self.warper.warp(faceDetection)
        facialHair = self.facialHairEstimator.estimate(warp)
        self.assertEstimation(facialHair, expectedResult)

    def test_async_estimate_glasses(self):
        """
        Test async estimate facial hair
        """
        faceDetection = self.detector.detectOne(WARP_NO_FACIAL_HAIR)
        warp = self.warper.warp(faceDetection)
        task = self.facialHairEstimator.estimate(warp.warpedImage, asyncEstimate=True)
        self.assertAsyncEstimation(task, FacialHair)
        task = self.facialHairEstimator.estimateBatch([warp.warpedImage] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, FacialHair)
