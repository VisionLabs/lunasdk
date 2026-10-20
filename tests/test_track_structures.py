"""Track structures built from synthetic tracker results (no FaceEngine data needed)."""

from collections import Counter

import numpy
import pytest
from FaceEngine import Detection, RectFloat, Vector2f
from FaceEngine.TrackEngine import BodyTrackData, FaceTrackData

from lunavl.sdk.detectors.bodydetector import BodyDetection
from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.trackengine.structures import BodyTrack, FaceTrack

IMAGE = VLImage(numpy.zeros((480, 640, 3), dtype=numpy.uint8))


class CountingProxy:
    """Proxy over a core track structure that counts attribute reads (pybind11 round-trips)."""

    def __init__(self, core):
        self._core = core
        self.reads: Counter = Counter()

    def __getattr__(self, name):
        self.reads[name] += 1
        return getattr(self._core, name)


def makeDetection():
    detection = Detection()
    detection.setRawRect(RectFloat(100.0, 120.0, 80.0, 100.0))
    detection.setScore(0.9)
    return detection


def makeFaceData(isDetector, points=()):
    data = FaceTrackData()
    data.detection = makeDetection()
    data.isDetector = isDetector
    landmarks = data.landmarks
    for idx, x, y in points:
        landmarks[idx] = Vector2f(x, y)
    data.landmarks = landmarks
    return data


def makeBodyData(isDetector):
    data = BodyTrackData()
    data.detection = makeDetection()
    data.isDetector = isDetector
    return data


@pytest.mark.parametrize(
    "points,hasLandmarks",
    [
        ((), False),
        (((0, -0.0, -0.0),), False),
        (((2, 0.0, 3.0),), True),
        (((4, 1e-30, 0.0),), True),
        (tuple((idx, 110.0 + idx, 220.0 + idx) for idx in range(5)), True),
    ],
    ids=["all_zero", "negative_zero", "y_only", "tiny", "all_points"],
)
def test_face_track_landmarks(points, hasLandmarks):
    """All-zero tracker landmarks mean "not estimated", any non-zero point means estimated landmarks"""
    detection = FaceTrack(makeFaceData(isDetector=True, points=points), IMAGE).detection

    assert isinstance(detection, FaceDetection)
    assert detection.boundingBox.score == pytest.approx(0.9)
    if not hasLandmarks:
        assert detection.landmarks5 is None
        assert not detection.coreEstimation.landmarks5_opt.isValid()
    else:
        assert detection.landmarks5 is not None
        assert detection.coreEstimation.landmarks5_opt.isValid()
        expected = {idx: (x, y) for idx, x, y in points}
        for idx, point in enumerate(detection.landmarks5.points):
            assert (point.x, point.y) == pytest.approx(expected.get(idx, (0.0, 0.0)))


@pytest.mark.parametrize("trackCls,makeData", [(FaceTrack, makeFaceData), (BodyTrack, makeBodyData)])
def test_track_detection_cached(trackCls, makeData):
    """The detection is built once per track wrapper"""
    core = CountingProxy(makeData(isDetector=True))
    track = trackCls(core, IMAGE)

    detection = track.detection
    assert detection is not None
    assert track.detection is detection
    assert track.detection is detection
    assert core.reads["isDetector"] == 1
    assert core.reads["detection"] == 1


@pytest.mark.parametrize("trackCls,makeData", [(FaceTrack, makeFaceData), (BodyTrack, makeBodyData)])
def test_track_without_detector_detection(trackCls, makeData):
    """A tracker-only track has no honest detection; the C++ isDetector flag is read once"""
    core = CountingProxy(makeData(isDetector=False))
    track = trackCls(core, IMAGE)

    assert track.detection is None
    assert track.detection is None
    assert track.detection is None
    assert core.reads["isDetector"] == 1


def test_body_track_detection():
    """Body detection of a detector track"""
    detection = BodyTrack(makeBodyData(isDetector=True), IMAGE).detection

    assert isinstance(detection, BodyDetection)
    assert detection.boundingBox.score == pytest.approx(0.9)
    # a synthetic detection is not "valid" for FaceEngine, so only the raw rect is filled
    assert detection.coreEstimation.detection.getRawRect().width == pytest.approx(80.0)
