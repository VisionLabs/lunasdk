"""
Facial hair estimation example
"""

import asyncio
import pprint

from resources import EXAMPLE_1, EXAMPLE_3

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateFacialHair():
    """
    Facial hair estimation example.
    """
    image = VLImage.load(filename=EXAMPLE_1)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    facialHairEstimator = faceEngine.createFacialHairEstimator()

    pprint.pprint(facialHairEstimator.estimate(warp).asDict())

    image2 = VLImage.load(filename=EXAMPLE_3)
    faceDetection2 = detector.detectOne(image2)
    warp2 = warper.warp(faceDetection2)
    estimations = facialHairEstimator.estimateBatch([warp, warp2])
    pprint.pprint([estimation.asDict() for estimation in estimations])


async def asyncEstimateFacialHair():
    """
    Async facial hair estimation example.
    """
    image = VLImage.load(filename=EXAMPLE_3)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    facialHairEstimator = faceEngine.createFacialHairEstimator()

    facialHair = await facialHairEstimator.estimate(warp, asyncEstimate=True)
    pprint.pprint(facialHair.asDict())

    image2 = VLImage.load(filename=EXAMPLE_3)
    faceDetection2 = detector.detectOne(image2)
    warp2 = warper.warp(faceDetection2)

    task1 = facialHairEstimator.estimateBatch([warp], asyncEstimate=True)
    task2 = facialHairEstimator.estimateBatch([warp2], asyncEstimate=True)

    for task in (task1, task2):
        estimations = task.get()
        pprint.pprint([estimation.asDict() for estimation in estimations])


if __name__ == "__main__":
    estimateFacialHair()
    asyncio.run(asyncEstimateFacialHair())
