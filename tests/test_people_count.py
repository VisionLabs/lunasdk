import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.image_estimators.people_count import EstimationTargets, ImageForPeopleEstimation
from lunavl.sdk.faceengine.setting_provider import PeopleCountEstimatorType
from lunavl.sdk.image_utils.geometry import Point, Rect
from lunavl.sdk.image_utils.image import ColorFormat, VLImage
from tests.base import BaseTestClass
from tests.resources import CROWD_7_PEOPLE, CROWD_9_PEOPLE, IMAGE_WITH_TWO_FACES


class TestPeopleCount(BaseTestClass):
    """
    Test estimate people count
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.peopleCountEstimator = cls.faceEngine.createPeopleCountEstimatorV2(PeopleCountEstimatorType.PEOPLE_COUNT_V2)
        cls.crowd9People = VLImage.load(filename=CROWD_9_PEOPLE)
        cls.crowd7People = VLImage.load(filename=CROWD_7_PEOPLE)
        cls.badFormatImage = VLImage.load(filename=CROWD_7_PEOPLE, colorFormat=ColorFormat.B8G8R8)
        cls.outsideArea = ImageForPeopleEstimation(
            cls.crowd9People, Rect(100, 100, cls.crowd9People.rect.width, cls.crowd9People.rect.height)
        )
        cls.areaLargerImage = ImageForPeopleEstimation(
            cls.crowd9People, Rect(100, 100, cls.crowd9People.rect.width + 100, cls.crowd9People.rect.height + 100)
        )
        cls.areaOutsideImage = ImageForPeopleEstimation(
            cls.crowd9People,
            Rect(
                cls.crowd9People.rect.width,
                cls.crowd9People.rect.height,
                cls.crowd9People.rect.width + 100,
                cls.crowd9People.rect.height + 100,
            ),
        )

    def test_people_count_async(self):  # todo: change asserted values LUNA-6049
        """
        Test single image async estimation
        """
        peopleCount = self.peopleCountEstimator.estimate(self.crowd9People, asyncEstimate=True).get()
        assert peopleCount.count == 10

    def test_people_count_batch_async(self):  # todo: change asserted values LUNA-6049
        """
        Test batch async estimation
        """
        peopleCount = self.peopleCountEstimator.estimateBatch(
            [self.crowd9People, self.crowd7People], asyncEstimate=True
        ).get()
        assert [estimation.count for estimation in peopleCount] == [10, 8]

    def test_people_count(self):  # todo: change asserted values LUNA-6049
        """
        Test single image estimation
        """
        peopleCount = self.peopleCountEstimator.estimate(self.crowd7People)
        assert peopleCount.count == 8

    def test_people_count_batch(self):  # todo: change asserted values LUNA-6049
        """
        Test batch estimation
        """
        images = [self.crowd9People, VLImage.load(filename=IMAGE_WITH_TWO_FACES), self.crowd7People]
        peopleCount = self.peopleCountEstimator.estimateBatch(images)
        assert [estimation.count for estimation in peopleCount] == [10, 2, 8]

    def test_people_count_with_batch_invalid_input(self):
        """
        Test estimation with bad input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize.format("Invalid span size"))

    def test_people_count_with_bad_format_image(self):
        """
        Test estimation with unsupported image format
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.badFormatImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.InvalidImageFormat)

    def test_people_count_batch_with_bad_format_image(self):
        """
        Test batch estimation with unsupported image format
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([self.crowd9People, self.badFormatImage])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[1], LunaVLError.InvalidImageFormat)

    def test_people_count_with_area_outside(self):
        """
        Test estimation with area slightly outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.outsideArea)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.InvalidRect)

    def test_people_count_batch_with_area_outside(self):
        """
        Test batch estimation with area slightly outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([self.crowd7People, self.outsideArea])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[1], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_with_area_larger_image(self):
        """
        Test estimation with area is larger than image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.areaLargerImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_batch_with_area_larger_image(self):
        """
        Test batch estimation with area is larger than image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([self.crowd7People, self.areaLargerImage])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[1], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_with_area_outside_image(self):
        """
        Test estimation with area completely outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.areaOutsideImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_batch_with_area_outside_image(self):
        """
        Test batch estimation with area completely outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([self.crowd7People, self.areaOutsideImage])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[1], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_batch_with_area_without_people(self):  # todo: change asserted values LUNA-6049
        """
        Test estimation with not contain people area
        """
        areaWithoutPeople = ImageForPeopleEstimation(self.crowd9People, Rect(10, 10, 100, 100))
        peopleCount = self.peopleCountEstimator.estimateBatch([areaWithoutPeople, self.crowd7People])
        assert [estimation.count for estimation in peopleCount] == [0, 8]

    def test_people_count_with_invalid_area(self):
        """
        Test estimation with invalid rectangle
        """
        invalidRectImage = ImageForPeopleEstimation(self.crowd9People, Rect(0, 0, 0, 0))
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(invalidRectImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_with_invalid_core_rect(self):
        """
        Test estimation with invalid core rectangle
        """
        errorCoreRectImage = ImageForPeopleEstimation(self.crowd9People, Rect(0.1, 0.1, 0.1, 0.1))
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(errorCoreRectImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_coordinates(self):
        """
        Test coordinates
        """
        peopleCount = self.peopleCountEstimator.estimate(self.crowd7People)
        assert len(peopleCount.coordinates) == 8
        for point in peopleCount.coordinates:
            assert isinstance(point, Point)
            assert isinstance(point.x, int)
            assert isinstance(point.y, int)

    def test_targets(self):
        """
        Test targets param
        """
        peopleCount = self.peopleCountEstimator.estimate(self.crowd7People, estimationTargets=EstimationTargets.T1)
        assert len(peopleCount.coordinates) == 8
        peopleCount = self.peopleCountEstimator.estimate(self.crowd7People, estimationTargets=EstimationTargets.T2)
        assert len(peopleCount.coordinates) == 0

    def test_V1(self):
        """
        Test old people estimated api
        """
        peopleCountEstimator = self.faceEngine.createPeopleCountEstimator(PeopleCountEstimatorType.PEOPLE_COUNT_V2)
        peopleCount = peopleCountEstimator.estimate(self.crowd7People)
        assert peopleCount == 8
        peopleCount = peopleCountEstimator.estimateBatch([self.crowd7People])
        assert peopleCount[0] == 8

    def test_asDict(self):
        """
        Test asDict method
        """
        peopleCount = self.peopleCountEstimator.estimate(self.crowd7People, estimationTargets=EstimationTargets.T1)
        assert {"count": peopleCount.count, "coordinates": peopleCount.coordinates} == peopleCount.asDict()
