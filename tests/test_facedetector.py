import pytest

from lunavl.sdk.detectors.base import ImageForDetection
from lunavl.sdk.detectors.facedetector import FaceDetection, FaceDetector
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import ColorFormat, VLImage
from tests.detect_test_class import (
    AREA_LARGER_IMAGE,
    AREA_OUTSIDE_IMAGE,
    AREA_WITHOUT_FACE,
    GOOD_AREA,
    LARGE_AREA,
    OUTSIDE_AREA,
    SMALL_AREA,
    VLIMAGE_BAD_IMAGE,
    VLIMAGE_LARGE_IMAGE,
    VLIMAGE_ONE_FACE,
    VLIMAGE_SEVERAL_FACE,
    VLIMAGE_SMALL,
    FaceDetectTestClass,
)
from tests.resources import BAD_IMAGE, MANY_FACES, ONE_FACE, WARP_CLEAN_FACE
from tests.schemas import LANDMARKS5, REQUIRED_FACE_DETECTION, jsonValidator


class TestFaceDetector(FaceDetectTestClass):
    """
    Test of face detector.
    """

    #: Face detector with default detector type
    defaultDetector: FaceDetector

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.defaultDetector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)

    def test_image_detection_with_transfer_option(self):
        """
        Test structure image for detection
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)])
                self.assertFaceDetection(detection[0], VLIMAGE_ONE_FACE)
                assert 1 == len(detection)

    def test_check_landmarks_points(self):
        """
        Test validation landmarks points
        """
        detection = TestFaceDetector.defaultDetector.detectOne(image=VLIMAGE_ONE_FACE, detect68Landmarks=True)
        self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

        self.assertLandmarksPoints(detection.landmarks5.points)
        self.assertLandmarksPoints(detection.landmarks68.points)

    def test_landmarks_as_dict(self):
        """
        Test conversion landmarks to dictionary
        """
        currentLandmarks5 = TestFaceDetector.defaultDetector.detectOne(image=VLIMAGE_ONE_FACE).landmarks5.asDict()

        assert (
            jsonValidator(schema=LANDMARKS5).validate(currentLandmarks5) is None
        ), f"{currentLandmarks5} does not match with schema {LANDMARKS5}"

    def test_valid_bounding_box(self):
        """
        Test validate bounding box (rect and score)
        """
        detection = TestFaceDetector.defaultDetector.detectOne(image=VLIMAGE_ONE_FACE)
        self.assertBoundingBox(detection.boundingBox)
        detection = TestFaceDetector.defaultDetector.detect(images=[VLIMAGE_ONE_FACE])[0][0]
        self.assertBoundingBox(detection.boundingBox)

    def test_bounding_box_as_dict(self):
        """
        Test conversion bounding box to dictionary
        """
        boundingBox = TestFaceDetector.defaultDetector.detectOne(image=VLIMAGE_ONE_FACE).boundingBox.asDict()

        assert (
            jsonValidator(schema=REQUIRED_FACE_DETECTION).validate(boundingBox) is None
        ), f"{boundingBox} does not match with schema {REQUIRED_FACE_DETECTION}"

    def test_face_detection_as_dict(self):
        """
        Test conversion result face detection to dictionary
        """
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                detectAsDict = TestFaceDetector.defaultDetector.detectOne(
                    image=VLIMAGE_ONE_FACE,
                    detect5Landmarks=case.detect5Landmarks,
                    detect68Landmarks=case.detect68Landmarks,
                ).asDict()
                assert (
                    jsonValidator(schema=REQUIRED_FACE_DETECTION).validate(detectAsDict) is None
                ), f"{detectAsDict} does not match with schema {REQUIRED_FACE_DETECTION}"

    def test_detection_with_default_detector_type(self):
        """
        Test face detection with default detector type
        """
        for detectionFunction in ("detect", "detectOne"):
            with self.subTest(detectionFunction=detectionFunction):
                if detectionFunction == "detectOne":
                    detection = TestFaceDetector.defaultDetector.detectOne(image=VLIMAGE_ONE_FACE)
                else:
                    detection = TestFaceDetector.defaultDetector.detect(images=[VLIMAGE_ONE_FACE])[0]
                self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

    def test_detect_one_using_different_type_detector(self):
        """
        Test detection of one face using different type of detector
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

    def test_batch_detect_using_different_type_detector(self):
        """
        Test batch detection using different type of detector
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detect(images=[VLIMAGE_ONE_FACE])[0]
                self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

    @pytest.mark.skip("cannot get an error")
    def test_batch_detect_with_success_and_error(self):
        """
        Test batch detection with success and error using FACE_DET_V3 (there is no error with other detector)
        """
        badWarp = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))
        badWarp.coreImage = VLIMAGE_SMALL.coreImage
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detect(images=[VLIMAGE_ONE_FACE, VLIMAGE_BAD_IMAGE])
                self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError)
                assert len(exceptionInfo.value.context) == 2, "Expect two errors in exception context"
                self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.Ok)
                self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[1], LunaVLError.Internal)

    def test_detect_one_with_image_of_several_faces(self):
        """
        Test detection of one face with image of several faces
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_SEVERAL_FACE)
                self.assertFaceDetection(detection, VLIMAGE_SEVERAL_FACE)

    def test_detect_one_with_image_without_faces(self):
        """
        Test detection of one face with image without faces
        """
        imageWithoutFace = VLImage.load(filename=BAD_IMAGE)
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=imageWithoutFace)
                assert detection is None, detection

    def test_batch_detect_with_image_without_faces(self):
        """
        Test batch face detection with image without faces
        """
        imageWithoutFace = VLImage.load(filename=BAD_IMAGE)
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detect(images=[imageWithoutFace])
                assert 0 == len(detection[0])

    def test_detect_one_by_area_without_face(self):
        """
        Test detection of one face by area without face
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=AREA_WITHOUT_FACE)
                assert detection is None, detection

    def test_detect_one_by_area_with_face(self):
        """
        Test detection of one face by area with face
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)
                self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

    def test_batch_detect_with_image_of_several_faces(self):
        """
        Test batch face detection with image of several faces
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detect(images=[VLIMAGE_SEVERAL_FACE])
                self.assertFaceDetection(detection[0], VLIMAGE_SEVERAL_FACE)
                assert 1 == len(detection)
                assert 5 == len(detection[0])

    def test_batch_detect_of_multiple_images(self):
        """
        Test batch detection of multiple images
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detect(images=[VLIMAGE_SEVERAL_FACE, VLIMAGE_ONE_FACE])
                self.assertFaceDetection(detection[0], VLIMAGE_SEVERAL_FACE)
                self.assertFaceDetection(detection[1], VLIMAGE_ONE_FACE)
                assert 2 == len(detection)
                assert 5 == len(detection[0])
                assert 1 == len(detection[1])

    def test_get_landmarks_for_detect_one(self):
        """
        Test get and check landmark instances for detection of one face
        """
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                for detector in self.detectors:
                    with self.subTest(detectorType=detector.detectorType):
                        detection = detector.detectOne(
                            image=VLIMAGE_ONE_FACE,
                            detect68Landmarks=case.detect68Landmarks,
                            detect5Landmarks=case.detect5Landmarks,
                        )
                        self.assertDetectionLandmarks(
                            detection=detection, landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks
                        )

    def test_get_landmarks_for_batch_detect(self):
        """
        Test get and check landmark instances for batch detect
        """
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                for detector in self.detectors:
                    with self.subTest(detectorType=detector.detectorType):
                        detection = detector.detect(
                            images=[VLIMAGE_ONE_FACE],
                            detect68Landmarks=case.detect68Landmarks,
                            detect5Landmarks=case.detect5Landmarks,
                        )[0][0]
                        self.assertDetectionLandmarks(
                            detection=detection, landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks
                        )

    def test_batch_detect_limit(self):
        """
        Test checking detection limit for an image
        """
        imageWithManyFaces = VLImage.load(filename=MANY_FACES)
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detect(images=[imageWithManyFaces])[0]
                assert 5 == len(detection)

                detection = detector.detect(images=[imageWithManyFaces], limit=20)[0]
                if detector.detectorType.name == "FACE_DET_V3":
                    assert 20 == len(detection)
                else:
                    assert 19 == len(detection)

    def test_detect_limit_bad_param(self):
        """
        Test batch detection with negative limit number
        """
        imageWithManyFaces = VLImage.load(filename=MANY_FACES)
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(TypeError) as exceptionInfo:
                    detector.detect(
                        images=[ImageForDetection(image=imageWithManyFaces, detectArea=GOOD_AREA)], limit=-1
                    )
                assert isinstance(exceptionInfo.value, TypeError) is True, "expected TypeError"

    def test_detect_one_limit_bad_param(self):
        """
        Test batch detection with negative limit number
        """
        imageWithManyFaces = VLImage.load(filename=MANY_FACES)
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(TypeError) as exceptionInfo:
                    detector.detectOne(
                        image=ImageForDetection(image=imageWithManyFaces, detectArea=GOOD_AREA), limit=-1
                    )
                assert isinstance(exceptionInfo.value, TypeError) is True, "expected TypeError"

    def test_detect_one_invalid_image_format(self):
        """
        Test invalid image format detection
        """
        imageWithOneFaces = VLImage.load(filename=ONE_FACE, colorFormat=ColorFormat.B8G8R8)
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detectOne(image=imageWithOneFaces)
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidImageFormat.format("Invalid image format"))

    def test_batch_detect_invalid_image_format(self):
        """
        Test invalid image format detection
        """
        colorToImageMap = self.getColorToImageMap()
        allowedColorsForDetection = {ColorFormat.R8G8B8}
        for colorFormat in set(colorToImageMap) - allowedColorsForDetection:
            colorImage = colorToImageMap[colorFormat]
            for detector in self.detectors:
                with self.subTest(detectorType=detector.detectorType, colorFormat=colorFormat.name):
                    if all(
                        (
                            colorFormat in (ColorFormat.R8G8B8X8, ColorFormat.IR_X8X8X8),
                            detector.detectorType == DetectorType.FACE_DET_V3,
                        )
                    ):
                        detector.detect(images=[colorImage])
                    else:
                        with pytest.raises(LunaSDKException) as exceptionInfo:
                            detector.detect(images=[colorImage])
                        self.assertLunaVlError(
                            exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation.")
                        )
                        assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
                        self.assertReceivedAndRawExpectedErrors(
                            exceptionInfo.value.context[0], LunaVLError.InvalidImageFormat.format("Failed validation.")
                        )

    def test_batch_detect_by_area_without_face(self):
        """
        Test batch face detection by area without face
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detect(
                    images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=AREA_WITHOUT_FACE)]
                )
                assert 1 == len(detection)
                assert 0 == len(detection[0])

    def test_batch_detect_by_area_with_face(self):
        """
        Test batch face detection by area with face
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)])
                assert 1 == len(detection[0])
                self.assertFaceDetection(detection[0], VLIMAGE_ONE_FACE)

    def test_bad_area_detection(self):
        """
        Test detection of one face in area outside image
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=OUTSIDE_AREA)
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect.format("Invalid rectangle"))

    def test_batch_detect_in_area_outside_image(self):
        """
        Test batch detection in area outside image
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=OUTSIDE_AREA)])
                self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
                assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
                self.assertReceivedAndRawExpectedErrors(
                    exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
                )

    @pytest.mark.skip("unstable")
    def test_excessive_image_list_detection(self):
        """
        Test excessive image list detection
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            TestFaceDetector.defaultDetector.detect(images=[VLIMAGE_ONE_FACE] * 40)
        self.assertLunaVlError(exceptionInfo, LunaVLError.HighMemoryUsage)

    def test_detect_one_invalid_rectangle(self):
        """
        Test detection of one face with an invalid rect
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=Rect())
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect.format("Invalid rectangle"))

    def test_batch_detect_invalid_rectangle(self):
        """
        Test batch face detection with an invalid rect
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=Rect())])
                self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
                assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
                self.assertReceivedAndRawExpectedErrors(
                    exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
                )

    def test_match_detection_one_image(self):
        """
        Test match of values at different detections (detectOne and detect) with one image
        """
        for image in (VLIMAGE_ONE_FACE, VLIMAGE_SMALL):
            for detector in self.detectors:
                with self.subTest(detectorType=detector.detectorType):
                    detectOne = detector.detectOne(image=image, detect68Landmarks=True)
                    batchDetect = detector.detect(images=[image] * 3, detect68Landmarks=True)
                    for detection in batchDetect:
                        for face in detection:
                            assert face.boundingBox.asDict() == detectOne.boundingBox.asDict()
                            assert face.landmarks5.asDict() == detectOne.landmarks5.asDict()
                            assert face.landmarks68.asDict() == detectOne.landmarks68.asDict()

    def test_detect_one_large_image(self):
        """
        Test detection of one face from large image
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_LARGE_IMAGE)
                self.assertFaceDetection(detection, VLIMAGE_LARGE_IMAGE)

    def test_detect_one_by_large_area(self):
        """
        Test detection of one face by large area
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_LARGE_IMAGE, detectArea=LARGE_AREA)
                self.assertFaceDetection(detection, VLIMAGE_LARGE_IMAGE)

    def test_detect_one_by_small_area(self):
        """
        Test detection of one face by small area
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=SMALL_AREA)
                self.assertIsNone(detection)

    def test_detect_one_by_area_larger_image(self):
        """
        Test detection of one face by small area
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with self.assertRaises(expected_exception=LunaSDKException):
                    detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=AREA_LARGER_IMAGE)

    def test_detect_one_by_area_outside_image(self):
        """
        Test detection of one face by small area
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with self.assertRaises(expected_exception=LunaSDKException):
                    detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=AREA_OUTSIDE_IMAGE)

    def test_async_detect_face(self):
        """
        Test async detect face
        """
        task = self.defaultDetector.detectOne(VLIMAGE_ONE_FACE, asyncEstimate=True)
        self.assertAsyncEstimation(task, FaceDetection)
        task = self.defaultDetector.detect([VLIMAGE_ONE_FACE] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, FaceDetection)
