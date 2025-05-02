import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.image_estimators.image_modification import ModificationStatus
from lunavl.sdk.image_utils.image import ColorFormat, VLImage
from tests.base import BaseTestClass
from tests.resources import ROTATED0, STATUE


class TestImageModification(BaseTestClass):
    """
    Test estimate image modification
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.imageModificationEstimator = cls.faceEngine.createImageModificationEstimator()
        cls.image1 = VLImage.load(filename=STATUE)
        cls.image2 = VLImage.load(filename=ROTATED0)

    def test_modification_async(self):
        """
        Test single image async estimation
        """
        imageModification = self.imageModificationEstimator.estimate(self.image1, asyncEstimate=True).get()
        assert imageModification.status == ModificationStatus.Unmodified

    def test_modification_batch_async(self):
        """
        Test batch async estimation
        """
        imageModifications = self.imageModificationEstimator.estimateBatch(
            [self.image1, self.image2], asyncEstimate=True
        ).get()
        assert [estimation.status for estimation in imageModifications] == [
            ModificationStatus.Unmodified,
            ModificationStatus.Modified,
        ]

    def test_modification(self):
        """
        Test single image estimation
        """
        imageModification = self.imageModificationEstimator.estimate(self.image1)
        assert imageModification.score > 0
        assert imageModification.status == ModificationStatus.Unmodified

    def test_modification_batch(self):
        """
        Test batch estimation
        """
        imageModifications = self.imageModificationEstimator.estimateBatch([self.image1, self.image2])
        assert [estimation.status for estimation in imageModifications] == [
            ModificationStatus.Unmodified,
            ModificationStatus.Modified,
        ]
        assert imageModifications[0].score != imageModifications[1].score

    def test_modification_with_batch_invalid_input(self):
        """
        Test estimation with bad input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.imageModificationEstimator.estimateBatch([])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize.format("Invalid span size"))

    @pytest.mark.skip("FSDK-5505")
    def test_modification_with_bad_format_image(self):
        """
        Test estimation with unsupported image format
        """
        badFormatImage = VLImage.load(filename=STATUE, colorFormat=ColorFormat.B8G8R8)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.imageModificationEstimator.estimate(badFormatImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.InvalidImageFormat)

    @pytest.mark.skip("FSDK-5505")
    def test_modification_batch_with_bad_format_image(self):
        """
        Test batch estimation with unsupported image format
        """
        badFormatImage = VLImage.load(filename=STATUE, colorFormat=ColorFormat.B8G8R8)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.imageModificationEstimator.estimateBatch([self.image1, badFormatImage])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[1], LunaVLError.InvalidImageFormat)

    def test_asDict(self):
        """
        Test asDict method
        """
        imageModification = self.imageModificationEstimator.estimate(self.image1)
        assert {
            "score": imageModification.score,
            "status": imageModification.status.value,
        } == imageModification.asDict()
