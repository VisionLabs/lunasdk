import unittest

from lunavl.sdk.estimators.face_estimators.facial_hair import FacialHair, FacialHairEstimator
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
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

    def validate_facial_dict(self, receivedDict: dict, expectedResults: str):
        """
        Validate facial hair dict
        """
        assert {"predominant_facial_hair", "estimations"} == receivedDict.keys()
        assert {"beard", "mustache", "no_hair", "stubble"} == receivedDict["estimations"].keys()
        for estimation, estimationValue in receivedDict["estimations"].items():
            assert 0 <= estimationValue <= 1
        self.assertEqual(receivedDict["predominant_facial_hair"], expectedResults)

    def assertEstimation(self, facialHair: FacialHair, expectedEstimation: str):
        """
        Function checks if the instance belongs to the FacialHair class and compares the result with what is expected.

        Args:
            facialHair: facial hair estimation object
            expectedEstimationResults: dictionary with result
        """
        assert isinstance(facialHair, FacialHair), f"{facialHair.__class__} is not {FacialHair}"

    def test_estimate_no_hair(self):
        """
        Test facial hair estimations without facial hair on the face
        """
        expectedResult = "no_hair"
        faceDetection = self.detector.detectOne(VLImage.load(filename=WARP_NO_FACIAL_HAIR))
        warp = self.warper.warp(faceDetection)
        facialHair = self.facialHairEstimator.estimate(warp).asDict()
        self.validate_facial_dict(facialHair, expectedResult)

    def test_estimate_beard(self):
        """
        Test facial hair estimation with beard on the face
        """
        expectedResult = "beard"
        faceDetection = self.detector.detectOne(VLImage.load(filename=WARP_FACE_WITH_BEARD))
        warp = self.warper.warp(faceDetection)
        facialHair = self.facialHairEstimator.estimate(warp).asDict()
        self.validate_facial_dict(facialHair, expectedResult)

    def test_estimate_stubble(self):
        """
        Test facial hair estimations with stubble on the face
        """
        expectedResult = "stubble"
        faceDetection = self.detector.detectOne(VLImage.load(filename=WARP_FACE_WITH_STUBBLE))
        warp = self.warper.warp(faceDetection)
        facialHair = self.facialHairEstimator.estimate(warp).asDict()
        self.validate_facial_dict(facialHair, expectedResult)

    def test_estimate_mustache(self):
        """
        Test facial hair estimations with mustache on the face
        """
        expectedResult = "mustache"
        faceDetection = self.detector.detectOne(VLImage.load(filename=WARP_FACE_WITH_MUSTACHE))
        warp = self.warper.warp(faceDetection)
        facialHair = self.facialHairEstimator.estimate(warp).asDict()
        self.validate_facial_dict(facialHair, expectedResult)

    def test_async_estimate_facial_hair(self):
        """
        Test async estimate facial hair
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=WARP_NO_FACIAL_HAIR))
        warp = self.warper.warp(faceDetection)
        task = self.facialHairEstimator.estimate(warp.warpedImage, asyncEstimate=True)
        self.assertAsyncEstimation(task, FacialHair)
        task = self.facialHairEstimator.estimateBatch([warp.warpedImage] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, FacialHair)
