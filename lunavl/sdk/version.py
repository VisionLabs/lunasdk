from typing import NamedTuple

import FaceEngine as CoreFE  # pylint: disable=E0611,E0401


class SemVer(NamedTuple):
    """Semver version"""

    major: int
    minor: int
    patch: int


class Version(NamedTuple):
    # sdk version
    version: SemVer
    # build hash
    hash: str


def getVersion() -> Version:
    """
    Get sdk version
    """
    version = SemVer(*CoreFE.getVersionString()[len("fsdk_version: ") :].split("."))
    gitHash = CoreFE.getVersionHash()[len("fsdk_hash: ") :]
    return Version(version, gitHash)


# sdk version
SDK_VERSION = getVersion()
