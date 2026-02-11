from importlib.metadata import version

from lunavl.sdk.version import getVersion
from tests.base import BaseTestClass


class TestGetVersion(BaseTestClass):

    def test_get_version(self):
        """Test get sdk version"""
        sdkVersion = getVersion()
        isinstance(sdkVersion.hash, str)
        packageVersion = version("FaceEngine")
        semver = sdkVersion.version
        assert packageVersion.startswith(f"{semver.major}.{semver.minor}.{semver.patch}")
