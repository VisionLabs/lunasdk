import unittest
from importlib.metadata import version

from lunavl.sdk.version import getVersion
from tests.base import BaseTestClass


@unittest.skip(
    "LUNA-8410: temporarily skipped while the batch-warp bindings test wheel "
    "(faceengine-test-8410) is used instead of the official FaceEngine package; "
    "restore after the official release with the batch warp API"
)
class TestGetVersion(BaseTestClass):

    def test_get_version(self):
        """Test get sdk version"""
        sdkVersion = getVersion()
        isinstance(sdkVersion.hash, str)
        packageVersion = version("FaceEngine")
        semver = sdkVersion.version
        assert packageVersion.startswith(f"{semver.major}.{semver.minor}.{semver.patch}")
