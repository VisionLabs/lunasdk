from pathlib import Path

from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarp, FaceWarper
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE


class TestFaceWarper(BaseTestClass):
    """
    Test estimate face warp.
    """

    warper: FaceWarper

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.detection = cls.detector.detectOne(VLImage.load(filename=ONE_FACE))

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
