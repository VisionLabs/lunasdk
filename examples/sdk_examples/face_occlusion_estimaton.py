"""
Face occlusion estimation example
"""

import asyncio
import pprint

from resources import EXAMPLE_1, EXAMPLE_3

from lunavl.sdk.estimators.face_estimators.face_occlusion import WarpWithLandmarks
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateFaceOcclusion():
    """
    Face occlusion estimation example.
    """
    image = VLImage.load(filename=EXAMPLE_3)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    landMarks5Transformation = warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")

    faceOcclusionEstimator = faceEngine.createFaceOcclusionEstimator()

    warpWithLandmarks = WarpWithLandmarks(warp, landMarks5Transformation)
    pprint.pprint(faceOcclusionEstimator.estimate(warpWithLandmarks).asDict())

    image2 = VLImage.load(filename=EXAMPLE_3)
    faceDetection2 = detector.detectOne(image2)
    warp2 = warper.warp(faceDetection2)
    landMarks5Transformation2 = warper.makeWarpTransformationWithLandmarks(faceDetection2, "L5")

    warpWithLandmarksList = [
        WarpWithLandmarks(warp, landMarks5Transformation),
        WarpWithLandmarks(warp2, landMarks5Transformation2),
    ]

    estimations = faceOcclusionEstimator.estimateBatch(warpWithLandmarksList)
    pprint.pprint([estimation.asDict() for estimation in estimations])


async def asyncEstimateFaceOcclusion():
    """
    Async face occlusion estimation example.
    """
    image = VLImage.load(filename=EXAMPLE_3)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    landMarks5Transformation = warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")

    faceOcclusionEstimator = faceEngine.createFaceOcclusionEstimator()

    warpWithLandmarks = WarpWithLandmarks(warp, landMarks5Transformation)
    faceOcclusion = await faceOcclusionEstimator.estimate(warpWithLandmarks, asyncEstimate=True)
    pprint.pprint(faceOcclusion.asDict())

    image2 = VLImage.load(filename=EXAMPLE_3)
    faceDetection2 = detector.detectOne(image2)
    warp2 = warper.warp(faceDetection2)
    landMarks5Transformation2 = warper.makeWarpTransformationWithLandmarks(faceDetection2, "L5")

    task1 = faceOcclusionEstimator.estimateBatch(
        [WarpWithLandmarks(warp, landMarks5Transformation)], asyncEstimate=True
    )
    task2 = faceOcclusionEstimator.estimateBatch(
        [WarpWithLandmarks(warp2, landMarks5Transformation2)], asyncEstimate=True
    )

    for task in (task1, task2):
        estimations = task.get()
        pprint.pprint([estimation.asDict() for estimation in estimations])


if __name__ == "__main__":
    estimateFaceOcclusion()
    asyncio.run(asyncEstimateFaceOcclusion())
