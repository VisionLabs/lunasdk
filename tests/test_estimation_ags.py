import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.ags import AGSEstimator, ImageWithFaceDetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import CLEAN_ONE_FACE, ONE_FACE

EXPECTED_PRECISION = 10**-5


class TestBasicAttributes(BaseTestClass):
    """Test basic attributes."""

    estimator: AGSEstimator
    # first image
    image1: VLImage = VLImage.load(filename=ONE_FACE)
    # second image
    image2: VLImage = VLImage.load(filename=CLEAN_ONE_FACE)

    @classmethod
    def setUpClass(cls) -> None:
        """Load warps."""
        # estimator to call
        cls.estimator: AGSEstimator = BaseTestClass.faceEngine.createAGSEstimator()
        detector = VLFaceEngine().createFaceDetector(DetectorType.FACE_DET_V1)
        cls.detection1 = detector.detectOne(cls.image1)
        cls.detection2 = detector.detectOne(cls.image2)

    def test_correctness_with_image(self):
        """
        Test estimation correctness with image.
        """
        expectedAgs = 0.89521
        imageWithFaceDetection = ImageWithFaceDetection(self.image1, self.detection1.boundingBox)

        singleValue = self.estimator.estimate(imageWithFaceDetection=imageWithFaceDetection)
        batchValue = self.estimator.estimateBatch([imageWithFaceDetection])[0]
        assert type(singleValue) == type(batchValue)
        assert isinstance(singleValue, float)
        assert abs(expectedAgs - singleValue) < EXPECTED_PRECISION

    def test_correctness_with_detections(self):
        """
        Test estimation correctness with detections.
        """
        expectedAgs = 0.89521
        singleValue = self.estimator.estimate(detection=self.detection1)
        batchValue = self.estimator.estimateBatch(detections=[self.detection1])[0]
        assert type(singleValue) == type(batchValue)
        assert isinstance(singleValue, float)
        assert abs(expectedAgs - singleValue) < EXPECTED_PRECISION

    def test_batch_with_images(self):
        """
        Test batch estimation correctness with images.
        """
        expectedAgsList = [0.89521, 0.91636]
        result = self.estimator.estimateBatch(
            [
                ImageWithFaceDetection(self.image1, self.detection1.boundingBox),
                ImageWithFaceDetection(self.image2, self.detection2.boundingBox),
            ]
        )
        assert isinstance(result, list)
        for idx, row in enumerate(result):
            assert isinstance(row, float)
            assert abs(row - expectedAgsList[idx]) < EXPECTED_PRECISION

    def test_batch_with_detections(self):
        """
        Test batch estimation correctness with detections.
        """
        expectedAgsList = [0.89521, 0.91636]
        result = self.estimator.estimateBatch(detections=[self.detection1, self.detection2])
        assert isinstance(result, list)
        for idx, row in enumerate(result):
            assert isinstance(row, float)
            assert abs(row - expectedAgsList[idx]) < EXPECTED_PRECISION

    def test_batch_with_detections_bad_input(self):
        """
        Test batch estimation with invalid input.
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.estimator.estimateBatch([])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize.format("Invalid span size"))
