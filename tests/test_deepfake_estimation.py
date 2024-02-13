"""
Test estimate deepfake.
"""

from lunavl.sdk.estimators.face_estimators.deepfake import Deepfake, DeepfakeEstimationMode, DeepfakePrediction
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
        cls.deepfakeEstimator = cls.faceEngine.createDeepfakeEstimator()

    def test_estimate_deepfake_correctness(self):
        """
        Test deepfake estimator correctness
        """
        images = {
            DeepfakePrediction.Fake: DEEPFAKE,
            DeepfakePrediction.Real: ONE_FACE,
        }
        for deepfakeState, image in images.items():
            with self.subTest(deepfakeState):
                estimation = self.estimate(image)
                assert deepfakeState == estimation.prediction
                assert estimation.asDict()["prediction"] == deepfakeState.name.lower()

    def estimate(self, image: str = ONE_FACE) -> Deepfake:
        """Estimate deepfake on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        estimation = self.deepfakeEstimator.estimate(faceDetection)
        assert isinstance(estimation, Deepfake)
        return estimation

    def test_estimate_deepfake(self):
        """
        Simple deepfake estimation
        """
        estimation = self.estimate(ONE_FACE)
        assert estimation.prediction == DeepfakePrediction.Real

    def test_estimate_as_dict(self):
        """
        Test method DeepFae.asDict
        """
        estimation = self.estimate(ONE_FACE)
        assert {"prediction": "real", "score": estimation.score} == estimation.asDict()

    def test_estimate_deepfake_batch(self):
        """
        Batch deepfake estimation test
        """
        faceDetections = self.detector.detect([VLImage.load(filename=ONE_FACE), VLImage.load(filename=DEEPFAKE)])

        estimations = self.deepfakeEstimator.estimateBatch(faceDetections[0] + faceDetections[1])
        assert DeepfakePrediction.Real == estimations[0].prediction
        assert DeepfakePrediction.Fake == estimations[1].prediction

    def test_estimate_deepfake_mode(self):
        """
        Deepfake estimation mode
        """

        deepFakeEstimator1 = self.faceEngine.createDeepfakeEstimator(mode=DeepfakeEstimationMode.M1)
        deepFakeEstimator2 = self.faceEngine.createDeepfakeEstimator(mode=DeepfakeEstimationMode.M2)
        faceDetection = self.detector.detectOne(VLImage.load(filename=DEEPFAKE))
        defaultEstimation = self.deepfakeEstimator.estimate(faceDetection)
        estimation1 = deepFakeEstimator1.estimate(faceDetection)
        estimation2 = deepFakeEstimator2.estimate(faceDetection)
        assert defaultEstimation.asDict() == estimation2.asDict()
        assert defaultEstimation.asDict() != estimation1.asDict()

    def test_async_detect_human(self):
        """
        Test async estimate deep fake feature
        """
        faceDetections = self.detector.detect([VLImage.load(filename=ONE_FACE), VLImage.load(filename=DEEPFAKE)])

        task = self.deepfakeEstimator.estimateBatch(faceDetections[0] + faceDetections[1], asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, Deepfake)
        estimations = task.get()
        assert DeepfakePrediction.Real == estimations[0].prediction
        assert DeepfakePrediction.Fake == estimations[1].prediction
        task = self.deepfakeEstimator.estimate(faceDetections[0][0], asyncEstimate=True)
        estimation = task.get()
        assert isinstance(estimation, Deepfake)
