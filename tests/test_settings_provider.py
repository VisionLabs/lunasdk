from lunavl.sdk.faceengine.setting_provider import FaceEngineSettingsProvider
from tests.base import BaseTestClass


class TestSettingsProvider(BaseTestClass):
    """Test Settings Providers"""

    def test_FaceDetV3Settings_ScoreThreshold(self):
        """
        Test check provide setting FaceDetV3::Settings.ScoreThreshold form faceengine.conf
        """
        setting = FaceEngineSettingsProvider().faceDetV3Settings.scoreThreshold
        assert setting is not None, f"FaceDetV3::Settings.ScoreThreshold is not provide from faceengine.conf"

    def test_BodyDetectorSettings_ScoreThreshold(self):
        """
        Test check provide setting HumanDetector::Settings.ScoreThreshold form faceengine.conf
        """
        setting = FaceEngineSettingsProvider().bodyDetectorSettings.scoreThreshold
        assert setting is not None, f"HumanDetector::Settings.ScoreThreshold is not provide from faceengine.conf"
