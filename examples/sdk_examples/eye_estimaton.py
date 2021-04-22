"""
Eyes estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_1


def estimateEyes():
    """
    Estimate emotion from a warped image.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    landMarks5Transformation = warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")

    eyesEstimator = faceEngine.createEyeEstimator()

    pprint.pprint(eyesEstimator.estimate(landMarks5Transformation, warp.warpedImage).asDict())

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetection2 = detector.detectOne(image2)
    warp2 = warper.warp(faceDetection2)
    landMarks5Transformation2 = warper.makeWarpTransformationWithLandmarks(faceDetection2, "L5")

    estimations = eyesEstimator.estimateBatch(
        [landMarks5Transformation, landMarks5Transformation2], [warp.warpedImage, warp2.warpedImage]
    )
    pprint.pprint([estimation.asDict() for estimation in estimations])


if __name__ == "__main__":
    estimateEyes()
