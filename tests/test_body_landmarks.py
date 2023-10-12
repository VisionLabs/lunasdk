from itertools import chain

import pytest

from lunavl.sdk.detectors.bodydetector import BodyDetection, BodyDetector, Landmarks17
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.body_estimators.landmarks import BodyLandmarksEstimator, _prepareBatch
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import IMAGE_WITH_TWO_FACES, ONE_FACE, SEVERAL_FACES


class TestBodyLandmarks(BaseTestClass):
    """
    Test body landmarks estimations.
    """

    #: Body detector
    detector: BodyDetector
    #: head pose estimator
    estimator: BodyLandmarksEstimator
    #: default image
    image: VLImage
    #: detection on default image
    detection: BodyDetection

    @classmethod
    def setup_class(cls):
        """
        Set up a data for tests.
        Create detection for estimations.
        """
        super().setup_class()
        cls.detector = cls.faceEngine.createBodyDetector()
        cls.estimator = cls.faceEngine.createBodyLandmarksEstimator()
        cls.image = VLImage.load(filename=ONE_FACE)
        cls.detection = cls.detector.detectOne(cls.image)

    def assertLandmarks17(self, landmarks: Landmarks17):
        """
        Assert landmarks17 estimation.

        Args:
            landmarks: an estimate landmarks17
        """
        assert isinstance(landmarks, Landmarks17), f"bad landmarks instance, type of landmarks is {type(landmarks)}"
        assert 17 == len(landmarks.points)

    def test_init_estimator(self):
        """
        Test init estimator.
        """
        estimator = self.faceEngine.createBodyLandmarksEstimator()
        assert isinstance(
            estimator, BodyLandmarksEstimator
        ), f"bad estimator instance, type of estimator  is {type(estimator)}"

    def test_estimate_landmarks17(self):
        """
        Estimating landmarks17.
        """
        detection = self.detector.detectOne(self.image, detectLandmarks=True)
        landmarks = self.estimator.estimate(self.detection)
        self.assertLandmarks17(landmarks)
        assert detection.landmarks17.asDict() == landmarks.asDict()

    def test_estimate_landmarks17_batch(self):
        """
        Estimating landmarks17.
        """
        detection = self.detector.detectOne(self.image, detectLandmarks=True)
        landmarks = self.estimator.estimate(self.detection)
        self.assertLandmarks17(landmarks)
        assert detection.landmarks17.asDict() == landmarks.asDict()

    def test_async_estimate_landmarks17(self):
        """
        Async estimating landmarks17.
        """
        task = self.estimator.estimate(self.detection, asyncEstimate=True)
        self.assertAsyncEstimation(task, Landmarks17)
        task = self.estimator.estimateBatch([self.detection] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, Landmarks17)

    def test_estimate_landmarks_batch(self):
        """
        Estimating landmarks17. Test correctness and order returning values.
        """
        image1 = VLImage.load(filename=SEVERAL_FACES)
        image2 = self.image
        image3 = VLImage.load(filename=IMAGE_WITH_TWO_FACES)
        bodyDetections = list(chain(*self.detector.detect([image1, image2, image3], detectLandmarks=True)))

        testCases = bodyDetections.copy(), bodyDetections.copy()[::2] + bodyDetections.copy()[1::2]
        for index, testCase in enumerate(testCases):
            with self.subTest(testCaseIdx=index):
                estimations = self.estimator.estimateBatch(testCase)
                for idx, landmarks in enumerate(estimations):
                    self.assertLandmarks17(landmarks)
                    assert testCase[idx].landmarks17.asDict() == landmarks.asDict()

    def test_batch_invalid_input(self):
        """
        Batch estimation invalid input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.estimator.estimateBatch([], [])
        self.assertLunaVlError(exceptionInfo, LunaVLError.ValidationFailed.format("Invalid span size"))

    def test_prepare_batch_image_aggregation(self):
        """
        Unittest for `_prepareBatch` function.  Check correctness detection aggregation by image.
        """
        image1 = VLImage.load(filename=SEVERAL_FACES)
        image2 = self.image
        image3 = VLImage.load(filename=IMAGE_WITH_TWO_FACES)
        res = self.detector.detect([image1, image2, image3], detectLandmarks=True)

        detections = {id(image1): [], id(image2): [], id(image3): []}
        allDetections = []
        for imageDetections in res:
            for detection in imageDetections:
                detections[id(imageDetections[0].image)].append(detection)
                allDetections.append(detection)
        testCases = allDetections.copy(), allDetections.copy()[::2] + allDetections.copy()[1::2]
        for index, testCase in enumerate(testCases):
            with self.subTest(index):
                preparedBatch = _prepareBatch(testCase)
                assert 3 == len(preparedBatch)
                for coreImage, coreDetections, originalIdx in preparedBatch:
                    for image in (image1, image2, image3):
                        if image.coreImage == coreImage:
                            break
                    else:
                        self.fail("original image not found")
                    assert len(detections[id(image)]) == len(coreDetections)
                    assert set([id(detection.coreEstimation.detection) for detection in detections[id(image)]]) == set(
                        [id(detection) for detection in coreDetections]
                    )
