"""
Deepfake estimation example
"""
import asyncio
import pprint

from resources import EXAMPLE_1, EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateDeepfake():
    """
    Estimate deep fake feature.
    """

    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)

    deepfakeEstimator = faceEngine.createDeepfakeEstimator()

    pprint.pprint(deepfakeEstimator.estimate(faceDetection).asDict())

    faceDetection2 = detector.detectOne(VLImage.load(filename=EXAMPLE_1), detect68Landmarks=True)
    pprint.pprint(deepfakeEstimator.estimateBatch([faceDetection, faceDetection2]))


async def asyncEstimateDeepfake():
    """
    Async estimate  deep fake feature.
    """

    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image, detect68Landmarks=True)

    deepFakeEstimator = faceEngine.createDeepfakeEstimator()

    deepFake = await deepFakeEstimator.estimate(faceDetection, asyncEstimate=True)
    pprint.pprint(deepFake.asDict())

    faceDetection2 = detector.detectOne(VLImage.load(filename=EXAMPLE_1), detect68Landmarks=True)
    task1 = deepFakeEstimator.estimateBatch([faceDetection, faceDetection], asyncEstimate=True)
    task2 = deepFakeEstimator.estimateBatch([faceDetection, faceDetection2], asyncEstimate=True)

    for task in (task1, task2):
        pprint.pprint(task.get())


if __name__ == "__main__":
    estimateDeepfake()
    asyncio.run(asyncEstimateDeepfake())
