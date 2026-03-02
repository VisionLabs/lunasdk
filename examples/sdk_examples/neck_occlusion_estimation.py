"""
Neck occlusion estimation example
"""

import asyncio
import pprint

from resources import EXAMPLE_1, EXAMPLE_3

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateNeckOcclusion():
    """
    Neck occlusion estimation example.
    """
    image = VLImage.load(filename=EXAMPLE_1)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    neckOcclusionEstimator = faceEngine.createNeckOcclusionEstimator()

    pprint.pprint(neckOcclusionEstimator.estimate(faceDetection).asDict())

    image2 = VLImage.load(filename=EXAMPLE_3)
    faceDetection2 = detector.detectOne(image2)
    estimations = neckOcclusionEstimator.estimateBatch([faceDetection, faceDetection2])
    pprint.pprint([estimation.asDict() for estimation in estimations])


async def asyncEstimateNeckOcclusion():
    """
    Async neck occlusion estimation example.
    """
    image = VLImage.load(filename=EXAMPLE_3)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)

    neckOcclusionEstimator = faceEngine.createNeckOcclusionEstimator()

    neckOcclusion = await neckOcclusionEstimator.estimate(faceDetection, asyncEstimate=True)
    pprint.pprint(neckOcclusion.asDict())

    image2 = VLImage.load(filename=EXAMPLE_3)
    faceDetection2 = detector.detectOne(image2)

    task1 = neckOcclusionEstimator.estimateBatch([faceDetection], asyncEstimate=True)
    task2 = neckOcclusionEstimator.estimateBatch([faceDetection2], asyncEstimate=True)

    for task in (task1, task2):
        estimations = task.get()
        pprint.pprint([estimation.asDict() for estimation in estimations])


if __name__ == "__main__":
    estimateNeckOcclusion()
    asyncio.run(asyncEstimateNeckOcclusion())
