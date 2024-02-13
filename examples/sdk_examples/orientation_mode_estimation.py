"""
Module realize simple examples following features:
    * orientation mode estimation
"""

import pprint

from resources import EXAMPLE_1, EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage


def estimateOrientationMode():
    """
    Example of a orientation mode estimation.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    orientationModeEstimator = faceEngine.createOrientationModeEstimator()
    #: estimate
    pprint.pprint(orientationModeEstimator.estimate(image))

    image2 = VLImage.load(filename=EXAMPLE_1)
    #: estimate batch
    pprint.pprint(orientationModeEstimator.estimateBatch([image, image2]))


if __name__ == "__main__":
    estimateOrientationMode()
