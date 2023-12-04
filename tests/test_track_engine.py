import numpy

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.trackengine.engine import VLTrackEngine
from lunavl.sdk.trackengine.setting_provider import TrackEngineSettingsProvider
from lunavl.sdk.trackengine.structures import Frame, StreamParams
from tests.resources import ONE_FACE


def shiftImage(image: VLImage, x, y):
    non = lambda s: s if s < 0 else None
    mom = lambda s: max(0, s)

    npImage = image.asNPArray()
    shiftImage = numpy.zeros_like(npImage)
    shiftImage[mom(y) : non(y), mom(x) : non(x)] = npImage[mom(-y) : non(-y), mom(-x) : non(-x)]
    return VLImage(shiftImage)


def teFabric(detectFace=1, detectBody=0):
    faceEngine = VLFaceEngine()
    trackEngineProvider = TrackEngineSettingsProvider()
    trackEngineProvider.bodyTracking.useBodyReid = False
    trackEngineProvider.detectors.useFaceDetector = detectFace
    trackEngineProvider.detectors.useBodyDetector = detectBody
    return VLTrackEngine(faceEngine, trackEngineConf=trackEngineProvider)


def test_simple_te_usage():
    """Simple test that tracking work"""
    te = teFabric()
    img = VLImage.load(filename=ONE_FACE)
    streamId = te.registerStream()
    track = []
    for i in range(10):
        frame = Frame(image=shiftImage(img, 10 * i, 10 * i), streamId=streamId, frameNumber=i)
        res = te.track([frame])
        track.append(res[0])

    trackIds = [tr.humanTracks[0].trackId for tr in track]
    assert set(trackIds) == {0}
    firstDetection = track[0].humanTracks[0].face.bbox
    prevDetection = firstDetection
    for idx, trackDetection in enumerate(track[1:]):
        assert trackDetection.humanTracks[0].face is not None, idx
        detection = trackDetection.humanTracks[0].face.bbox

        assert abs(detection.center.x - prevDetection.center.x) < 50, idx + 1
        assert abs(detection.center.y - prevDetection.center.y) < 50, idx + 1
        prevDetection = detection
    assert abs(firstDetection.center.x - prevDetection.center.x) > 70
    assert abs(firstDetection.center.y - prevDetection.center.y) > 70
    tracks = te.closeStream(streamId)
    assert tracks == []


def test_te_roi():
    """Test that roi work"""
    te = teFabric()
    img = VLImage.load(filename=ONE_FACE)
    roiWithoutFace = Rect(0.6, 0.6, 0.1, 0.1)
    roiWithFace = Rect(0.1, 0.1, 0.6, 0.6)
    streamParams = StreamParams(roi=roiWithoutFace)
    streamId1 = te.registerStream(streamParams)
    streamParams = StreamParams(roi=roiWithFace)
    streamId2 = te.registerStream(streamParams)
    frame1 = Frame(image=img, streamId=streamId1, frameNumber=0)
    frame2 = Frame(image=img, streamId=streamId2, frameNumber=0)
    res = te.track([frame1, frame2])
    assert res[0].humanTracks == []
    assert res[1].humanTracks != []


def test_stream_params():
    pass


def test_detector_types():
    pass


def test_close_stream():
    pass


def test_async_stream():
    pass
