"""
Module realize simple fisheye estimation examples.
"""

import pprint

from resources import EXAMPLE_1, EXAMPLE_O

from lunavl.sdk.estimators.base import ImageWithFaceDetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateFisheye():
    """
    Example of a fisheye estimation.

    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    warper = faceEngine.createFaceWarper()
    fishEstimator = faceEngine.createFisheyeEstimator()
    faceDetection = detector.detectOne(image)
    warp = warper.warp(faceDetection)

    #: single estimation
    fisheye = fishEstimator.estimate(warp)
    pprint.pprint(fisheye)

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetection2 = detector.detectOne(image2)
    warp = warper.warp(faceDetection2)

    #: batch estimation
    fisheyeList = fishEstimator.estimateBatch([warp, warp2])
    pprint.pprint(fisheyeList)


if __name__ == "__main__":
    estimateFisheye()
