"""
Warps visualization example.
"""

import pprint

import cv2  # pylint: disable=E0611,E0401

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def createWarp():
    """
    Create warp from detection.

    """
    image = VLImage.load(
        url="https://img.championat.com/s/1350x900/news/big/g/v/oficialno-hbo-snimet-serial-garri-potter_16813235311040883005.jpg"
    )
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    pprint.pprint(warp.warpedImage.rect)
    cv2.imshow("Wapred image", warp.warpedImage.asNPArray())
    cv2.imshow("Original image", image.asNPArray())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    createWarp()
