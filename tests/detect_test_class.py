import itertools
from collections import namedtuple
from typing import List, Type, Union

from lunavl.sdk.base import BoundingBox
from lunavl.sdk.detectors.base import BaseDetection
from lunavl.sdk.detectors.bodydetector import BodyDetection, BodyDetector
from lunavl.sdk.detectors.facedetector import FaceDetection, FaceDetector, Landmarks5, Landmarks68
from lunavl.sdk.faceengine.engine import DetectorType
from lunavl.sdk.image_utils.geometry import Point, Rect
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import BAD_IMAGE, LARGE_IMAGE, ONE_FACE, SEVERAL_FACES, SMALL_IMAGE

VLIMAGE_SMALL = VLImage.load(filename=SMALL_IMAGE)
VLIMAGE_ONE_FACE = VLImage.load(filename=ONE_FACE)
VLIMAGE_BAD_IMAGE = VLImage.load(filename=BAD_IMAGE)
VLIMAGE_SEVERAL_FACE = VLImage.load(filename=SEVERAL_FACES)
VLIMAGE_LARGE_IMAGE = VLImage.load(filename=LARGE_IMAGE)
GOOD_AREA = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width - 100, VLIMAGE_ONE_FACE.rect.height - 100)
OUTSIDE_AREA = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width, VLIMAGE_ONE_FACE.rect.height)
SMALL_AREA = Rect(50, 50, 52, 52)
LARGE_AREA = Rect(1, 1, VLIMAGE_LARGE_IMAGE.rect.width - 1, VLIMAGE_LARGE_IMAGE.rect.height - 1)
AREA_LARGER_IMAGE = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width + 100, VLIMAGE_ONE_FACE.rect.height + 100)
AREA_OUTSIDE_IMAGE = Rect(
    VLIMAGE_ONE_FACE.rect.width,
    VLIMAGE_ONE_FACE.rect.height,
    VLIMAGE_ONE_FACE.rect.width + 100,
    VLIMAGE_ONE_FACE.rect.height + 100,
)
AREA_WITHOUT_FACE = Rect(50, 50, 100, 100)
INVALID_RECT = Rect(0, 0, 0, 0)
ERROR_CORE_RECT = Rect(0.1, 0.1, 0.1, 0.1)  # anything out of range (0.1, 1)


class BaseDetectorTestClass(BaseTestClass):
    """
    Base class for detectors tests
    """

    #: detection class
    detectionClass: Type[BaseDetection]

    def assertBoundingBox(self, boundingBox: BoundingBox):
        """
        Assert attributes of Bounding box class

        Args:
            boundingBox: bounding box
        """
        assert isinstance(boundingBox, BoundingBox), f"{boundingBox} is not {BoundingBox}"
        self.checkRectAttr(boundingBox.rect)

        assert isinstance(boundingBox.score, float), f"{boundingBox.score} is not float"
        assert 0 <= boundingBox.score < 1, "score out of range [0,1]"

    @staticmethod
    def assertPoint(point: Point):
        """
        Assert landmark point
        Args:
            point: point
        """
        assert isinstance(point, Point), "Landmarks does not contains Point"
        assert isinstance(point.x, float) and isinstance(point.y, float), "point coordinate is not float"

    def assertDetection(
        self,
        detection: Union[FaceDetection, BodyDetection, Union[List[FaceDetection], List[BodyDetection]]],
        imageVl: VLImage,
    ):
        """
        Function checks if an instance is Detection class, image and bounding box

        Args:
            detection: detection
            imageVl: class image
        """
        if isinstance(detection, list):
            detectionList = detection
        else:
            detectionList = [detection]  # type: ignore

        for detection in detectionList:
            assert isinstance(detection, self.__class__.detectionClass), (
                f"{detection.__class__} is not " f"{self.__class__.detectionClass}"
            )
            assert detection.image.asPillow() == imageVl.asPillow(), "Detection image does not match VLImage"
            self.assertBoundingBox(detection.boundingBox)


class BodyDetectTestClass(BaseDetectorTestClass):
    """
    Base class for human body detection tests
    """

    #: global human detector
    detector: BodyDetector
    detectionClass: Type[BodyDetection] = BodyDetection

    @classmethod
    def setup_class(cls):
        """
        Create list of face detector
        """
        super().setup_class()
        cls.detector = cls.faceEngine.createBodyDetector()

    def assertBodyDetection(self, detection: Union[BodyDetection, List[BodyDetection]], imageVl: VLImage):
        """
        Function checks if an instance is FaceDetection class

        Args:
            detection: face detection
            imageVl: class image
        """
        self.assertDetection(detection, imageVl)


class FaceDetectTestClass(BaseDetectorTestClass):
    detectors: List[FaceDetector]
    detectionClass: Type[FaceDetection] = FaceDetection

    @classmethod
    def setup_class(cls):
        """
        Create list of face detector
        """
        super().setup_class()
        # no plans
        cls.detectors = [
            # cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V1),
            # cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V2),
            cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3),
        ]
        CaseLandmarks = namedtuple("CaseLandmarks", ("detect5Landmarks", "detect68Landmarks"))
        cls.landmarksCases = [
            CaseLandmarks(landmarks5, landmarks68)
            for landmarks5, landmarks68 in itertools.product((True, False), (True, False))
        ]

    def assertFaceDetection(self, detection: Union[FaceDetection, List[FaceDetection]], imageVl: VLImage):
        """
        Function checks if an instance is FaceDetection class

        Args:
            detection: face detection
            imageVl: class image
        """
        self.assertDetection(detection, imageVl)

    @staticmethod
    def assertDetectionLandmarks(
        detection: FaceDetection, landmarks5: Landmarks5 = None, landmarks68: Landmarks68 = None
    ):
        if landmarks5:
            assert isinstance(detection.landmarks5, Landmarks5), f"{detection.landmarks5.__class__} is not {Landmarks5}"
        else:
            assert detection.landmarks5 is None, detection.landmarks5
        if landmarks68:
            assert isinstance(
                detection.landmarks68, Landmarks68
            ), f"{detection.landmarks68.__class__} is not {Landmarks68}"
        else:
            assert detection.landmarks68 is None, detection.landmarks68

    @staticmethod
    def assertLandmarksPoints(landmarksPoints: tuple):
        """
        Assert landmarks points

        Args:
            landmarksPoints: tuple of landmarks points
        """
        assert isinstance(landmarksPoints, tuple), f"{landmarksPoints} points is not tuple"
        for point in landmarksPoints:
            BaseDetectorTestClass.assertPoint(point)
