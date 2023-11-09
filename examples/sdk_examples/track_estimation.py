"""
An mouth state estimation example
"""
import asyncio
import pprint

from lunavl.sdk.trackengine.engine import VLTrackEngine
from lunavl.sdk.trackengine.structures import Frame
from resources import EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateFaceTrack():
    """
    Estimate emotion from a warped image.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    trackengine = VLTrackEngine(faceEngine)
    streamId = trackengine.registerStream()
    res = trackengine.track([Frame(image, 1, streamId)])
    pprint.pprint(res)


# async def asyncEstimateMouthState():
#     """
#     Async mouth state estimation example.
#     """
#     image = VLImage.load(filename=EXAMPLE_O)
#     faceEngine = VLFaceEngine()
#     detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
#     faceDetection = detector.detectOne(image)
#     warper = faceEngine.createFaceWarper()
#     warp = warper.warp(faceDetection)
#
#     mouthEstimator = faceEngine.createMouthEstimator()
#     mouth = await mouthEstimator.estimate(warp.warpedImage, asyncEstimate=True)
#     pprint.pprint(mouth.asDict())
#
#     task1 = mouthEstimator.estimate(warp.warpedImage, asyncEstimate=True)
#     task2 = mouthEstimator.estimate(warp.warpedImage, asyncEstimate=True)
#
#     for task in (task1, task2):
#         pprint.pprint(task.get())
#

if __name__ == "__main__":
    estimateFaceTrack()
    # asyncio.run(asyncEstimateMouthState())
