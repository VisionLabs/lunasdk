import numpy
import pytest

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.trackengine.engine import VLTrackEngine
from lunavl.sdk.trackengine.setting_provider import TrackEngineSettingsProvider
from lunavl.sdk.trackengine.structures import Frame, StreamParams, HumanTrackingParams
from tests.resources import ONE_FACE


def shiftImage(image: VLImage, x, y):
    non = lambda s: s if s < 0 else None
    mom = lambda s: max(0, s)

    npImage = image.asNPArray()
    shiftImage = numpy.zeros_like(npImage)
    shiftImage[mom(y) : non(y), mom(x) : non(x)] = npImage[mom(-y) : non(-y), mom(-x) : non(-x)]
    return VLImage(shiftImage)


def teFabric(detectFace=1, detectBody=0, useBodyReid=0):
    faceEngine = VLFaceEngine()
    trackEngineProvider = TrackEngineSettingsProvider()
    trackEngineProvider.bodyTracking.useBodyReid = useBodyReid
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
    humanTrackingParams = HumanTrackingParams(
        inactiveTracksLifetime=4,
        iouConnectionThreshold=0.123,
        reIDMatchingDetectionsCount=4,
        reIDMatchingThreshold=0.123,
        removeOverlappedStrategy="SCORE",
        removeHorizontalRatio=0.123,
    )
    params = StreamParams(
        callbackBufferSize=4,
        detectorScaling=True,
        detectorStep=4,
        framesBufferSize=4,
        roi=Rect(0.1, 0.1, 0.6, 0.6),
        minimalTrackLength=4,
        killIntersectedIOUThreshold=0.123,
        scaledSize=4,
        skipFrames=4,
        trackingResultsBufferSize=4,
        useFrg=False,
        humanTrackingParams=humanTrackingParams,
    )

    assert params.callbackBufferSize == 4
    assert params.detectorScaling
    assert params.detectorStep == 4
    assert params.framesBufferSize == 4
    assert params.minimalTrackLength == 4
    assert pytest.approx(params.killIntersectedIOUThreshold) == 0.123
    assert params.scaledSize == 4
    assert params.skipFrames == 4
    assert params.trackingResultsBufferSize == 4
    assert not params.useFrg
    assert pytest.approx(params.killIntersectedIOUThreshold) == 0.123
    assert params.roi == Rect(0.1, 0.1, 0.6, 0.6)
    assert params.humanTrackingParams.inactiveTracksLifetime == 4
    assert params.humanTrackingParams.reIDMatchingDetectionsCount == 4
    assert pytest.approx(params.humanTrackingParams.iouConnectionThreshold) == 0.123
    assert pytest.approx(params.humanTrackingParams.reIDMatchingThreshold) == 0.123
    assert pytest.approx(params.humanTrackingParams.removeHorizontalRatio) == 0.123
    assert params.humanTrackingParams.removeOverlappedStrategy == "SCORE"


@pytest.mark.parametrize("detectFace,detectBody", [(1, 0), (0, 1), (1, 1)])
def test_detector_types(detectFace, detectBody):
    """Detector type test"""
    te = teFabric(detectFace=detectFace, detectBody=detectBody)
    img = VLImage.load(filename=ONE_FACE)
    streamId = te.registerStream()
    frame = Frame(image=img, streamId=streamId, frameNumber=1)
    res = te.track([frame])
    track = res[0].humanTracks[0]
    if detectFace:
        assert track.face is not None
    else:
        assert track.face is None
    if detectBody:
        assert track.body is not None
    else:
        assert track.body is None


@pytest.mark.parametrize("detectFace,detectBody", [(1, 0), (0, 1), (1, 1)])
def test_close_stream(detectFace, detectBody):
    te1 = teFabric(detectFace=detectFace, detectBody=detectBody, useBodyReid=1)
    te2 = teFabric(detectFace=detectFace, detectBody=detectBody, useBodyReid=0)
    img = VLImage.load(filename=ONE_FACE)
    streamId1 = te1.registerStream()
    streamId2 = te2.registerStream()
    frame1 = Frame(image=img, streamId=streamId1, frameNumber=1)
    frame2 = Frame(image=img, streamId=streamId2, frameNumber=1)
    res1 = te1.track([frame1])
    res2 = te2.track([frame2])
    if detectBody:
        assert res1 == []
        res = te1.closeStream(streamId1)
        assert len(res[0].humanTracks) > 0
        assert [frameRes.asDict() for frameRes in res] == [frameRes.asDict() for frameRes in res2]
    else:
        assert te1.closeStream(streamId1) == []
        assert [frameRes.asDict() for frameRes in res1] == [frameRes.asDict() for frameRes in res2]


def test_async_stream():
    te = teFabric()
    img = VLImage.load(filename=ONE_FACE)
    streamId = te.registerStream()
    frame = Frame(image=img, streamId=streamId, frameNumber=1)
    res = te.track([frame], asyncEstimate=True).get()
    assert len(res[0].humanTracks) > 0
