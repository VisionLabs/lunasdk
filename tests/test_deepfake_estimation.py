import snakecase

from lunavl.sdk.estimators.face_estimators.deepfake import DeepFake, DeepFakeEstimationMode, DeepFakeState
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import DEEPFAKE, ONE_FACE


class TestDeepfake(BaseTestClass):
    """
    Test estimate deepfake.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.deepFakeEstimator = cls.faceEngine.createDeepFakeEstimator()

    def test_estimate_deepfake_correctness(self):
        """
        Test deepfake estimator correctness
        """
        images = {
            DeepFakeState.Fake: DEEPFAKE,
            DeepFakeState.Real: ONE_FACE,
        }
        for deepfakeState, image in images.items():
            with self.subTest(deepfakeState):
                estimation = self.estimate(image)
                assert deepfakeState == estimation.state
                assert estimation.asDict()["state"] == deepfakeState.name.lower()

    def estimate(self, image: str = ONE_FACE) -> DeepFake:
        """Estimate deepfake on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        estimation = self.deepFakeEstimator.estimate(faceDetection)
        assert isinstance(estimation, DeepFake)
        return estimation

    def test_estimate_deepfake(self):
        """
        Simple deepfake estimation
        """
        estimation = self.estimate(ONE_FACE)
        assert estimation.state == DeepFakeState.Real

    def test_estimate_as_dict(self):
        """
        Test method DeepFae.asDict
        """
        estimation = self.estimate(ONE_FACE)
        assert {"state": "real", "score": estimation.score} == estimation.asDict()

    def test_estimate_deepfake_batch(self):
        """
        Batch deepfake estimation test
        """
        faceDetections = self.detector.detect([VLImage.load(filename=ONE_FACE), VLImage.load(filename=DEEPFAKE)])

        estimations = self.deepFakeEstimator.estimateBatch(faceDetections[0] + faceDetections[1])
        assert DeepFakeState.Real == estimations[0].state
        assert DeepFakeState.Fake == estimations[1].state

    def test_estimate_deepfake_mode(self):
        """
        Deepfake estimation mode
        """

        deepFakeEstimator1 = self.faceEngine.createDeepFakeEstimator(mode=DeepFakeEstimationMode.M1)
        deepFakeEstimator2 = self.faceEngine.createDeepFakeEstimator(mode=DeepFakeEstimationMode.M2)
        faceDetection = self.detector.detectOne(VLImage.load(filename=DEEPFAKE))
        defaultEstimation = self.deepFakeEstimator.estimate(faceDetection)
        estimation1 = deepFakeEstimator1.estimate(faceDetection)
        estimation2 = deepFakeEstimator2.estimate(faceDetection)
        assert defaultEstimation.asDict() == estimation2.asDict()
        assert defaultEstimation.asDict() != estimation1.asDict()

    def test_async_detect_human(self):
        """
        Test async estimate deep fake feature
        """
        faceDetections = self.detector.detect([VLImage.load(filename=ONE_FACE), VLImage.load(filename=DEEPFAKE)])

        task = self.deepFakeEstimator.estimateBatch(faceDetections[0] + faceDetections[1], asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, DeepFake)
        estimations = task.get()
        assert DeepFakeState.Real == estimations[0].state
        assert DeepFakeState.Fake == estimations[1].state
        task = self.deepFakeEstimator.estimate(faceDetections[0][0], asyncEstimate=True)
        estimation = task.get()
        assert isinstance(estimation, DeepFake)
