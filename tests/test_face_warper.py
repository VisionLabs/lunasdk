from pathlib import Path

from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarp, FaceWarper
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import CLEAN_ONE_FACE, ONE_FACE


class TestFaceWarper(BaseTestClass):
    """
    Test estimate face warp.
    """

    warper: FaceWarper
    detections: list

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.detection = cls.detector.detectOne(VLImage.load(filename=ONE_FACE))
        cls.detection_2 = cls.detector.detectOne(VLImage.load(filename=CLEAN_ONE_FACE))
        cls.detections = [cls.detection, cls.detection_2]

    def test_warp(self):
        """Test warp estimation"""
        warp = self.warper.warp(self.detection)
        assert isinstance(warp, FaceWarp)
        assert warp.warpedImage.filename == Path(ONE_FACE).name

    def test_warp_async(self):
        """Test async warp estimation"""
        warp = self.warper.warp(self.detection, asyncEstimate=True).get()
        assert isinstance(warp, FaceWarp)
        assert warp.warpedImage.filename == Path(ONE_FACE).name

    def test_warp_batch(self):
        """Test warp batch estimation"""
        warps = self.warper.warp_batch(self.detections)
        assert len(warps) == len(self.detections)
        for warp, detection in zip(warps, self.detections):
            assert isinstance(warp, FaceWarp)
            assert warp.sourceDetection == detection
            assert warp.warpedImage.filename == Path(detection.image.filename).name

        sequentialWarps = [self.warper.warp(detection) for detection in self.detections]
        assert len(warps) == len(sequentialWarps)

    def test_warp_batch_async(self):
        """Test async warp batch estimation"""
        warps = self.warper.warp_batch(self.detections, asyncEstimate=True).get()
        assert len(warps) == len(self.detections)
        for warp, detection in zip(warps, self.detections):
            assert isinstance(warp, FaceWarp)
            assert warp.sourceDetection == detection
            assert warp.warpedImage.filename == Path(detection.image.filename).name

    def test_warp_batch_validation(self):
        """Test warp batch with empty detections"""
        with self.assertRaises(ValueError):
            self.warper.warp_batch([])
