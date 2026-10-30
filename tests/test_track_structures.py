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


def filledSlots(obj) -> dict:
    """Slots of an object which are set, with their values"""
    return {
        slot: getattr(obj, slot)
        for cls in type(obj).__mro__
        for slot in getattr(cls, "__slots__", ())
        if hasattr(obj, slot)
    }


def rawRect(detection: Detection) -> tuple:
    rect = detection.getRawRect()
    return rect.x, rect.y, rect.width, rect.height


#: raw rect of `makeDetection`
TRACK_RAW_RECT = (100.0, 120.0, 80.0, 100.0)


def assertBoundingBoxIsCopy(detection, data, track) -> None:
    """The bbox comes from the tracker detection but is a copy: writing to it does not touch the tracker state"""
    boundingBox = detection.boundingBox.coreEstimation
    assert rawRect(boundingBox) == pytest.approx(TRACK_RAW_RECT)
    assert detection.boundingBox.score == pytest.approx(0.9)
    # the bbox is the detection of the core estimation itself, as `BaseDetection.__init__` makes it
    assert boundingBox is detection.coreEstimation.detection
    assert boundingBox is not data.detection
    boundingBox.setRawRect(RectFloat(1.0, 2.0, 3.0, 4.0))
    assert rawRect(data.detection) == pytest.approx(TRACK_RAW_RECT)
    assert rawRect(track.coreEstimation.detection) == pytest.approx(TRACK_RAW_RECT)


@pytest.mark.parametrize(
    "points", [(), tuple((idx, 110.0 + idx, 220.0 + idx) for idx in range(5))], ids=["no_landmarks", "landmarks"]
)
def test_face_track_detection_same_as_constructor(points):
    """The hot-path face detection is the same as one built by the FaceDetection constructor"""
    data = makeFaceData(isDetector=True, points=points)
    track = FaceTrack(data, IMAGE)
    detection = track.detection
    expected = FaceDetection(detection.coreEstimation, IMAGE)

    assert type(detection) is FaceDetection
    assert detection.image is IMAGE
    # a freshly built core face has no landmarks68, so `_fromTrackResult` does not read the optional
    assert not detection.coreEstimation.landmarks68_opt.isValid()
    assert detection.landmarks68 is None
    assert (detection.landmarks5 is None) == (expected.landmarks5 is None)
    # both paths fill the same slots, a field added to __init__ only fails here
    assert filledSlots(detection).keys() == filledSlots(expected).keys()
    assert detection.boundingBox.coreEstimation is expected.boundingBox.coreEstimation
    assert detection.asDict() == expected.asDict()
    assertBoundingBoxIsCopy(detection, data, track)


def test_body_track_detection_same_as_constructor():
    """The hot-path body detection is the same as one built by the BodyDetection constructor"""
    data = makeBodyData(isDetector=True)
    track = BodyTrack(data, IMAGE)
    detection = track.detection
    expected = BodyDetection(detection.coreEstimation, IMAGE)

    assert type(detection) is BodyDetection
    assert detection.image is IMAGE
    assert filledSlots(detection).keys() == filledSlots(expected).keys()
    assert detection.boundingBox.coreEstimation is expected.boundingBox.coreEstimation
    assert detection.asDict() == expected.asDict()
    assertBoundingBoxIsCopy(detection, data, track)
