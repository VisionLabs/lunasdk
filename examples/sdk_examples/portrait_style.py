"""
Module realize simple portraitStyle estimation examples.
"""

import pprint

from resources import EXAMPLE_1, EXAMPLE_O

from lunavl.sdk.estimators.base import ImageWithFaceDetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimatePortraitStyle():
    """
    Example of a portrait style estimation.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    estimator = faceEngine.createPortraitStyleEstimator()
    faceDetection = detector.detectOne(image)

    #: single estimation
    portraitStyle = estimator.estimate(faceDetection)
    pprint.pprint(portraitStyle)

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetection2 = detector.detectOne(image2)

    #: batch estimation
    portraitStyleList = estimator.estimateBatch([faceDetection, faceDetection2])
    pprint.pprint(portraitStyleList)


if __name__ == "__main__":
    estimatePortraitStyle()
