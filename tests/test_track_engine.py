import numpy
import pytest

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.trackengine.engine import VLTrackEngine
from lunavl.sdk.trackengine.setting_provider import TrackEngineSettingsProvider
from lunavl.sdk.trackengine.structures import Frame, HumanTrackingParams, StreamParams
from tests.resources import CLEAN_ONE_FACE, ONE_FACE


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

    def assertParams(savedParams):
        assert savedParams.callbackBufferSize == 4
        assert savedParams.detectorScaling
        assert savedParams.detectorStep == 4
        assert savedParams.framesBufferSize == 4
        assert savedParams.minimalTrackLength == 4
        assert pytest.approx(savedParams.killIntersectedIOUThreshold) == 0.123
        assert savedParams.scaledSize == 4
        assert savedParams.skipFrames == 4
        assert savedParams.trackingResultsBufferSize == 4
        assert not savedParams.useFrg
        assert pytest.approx(savedParams.killIntersectedIOUThreshold) == 0.123
        assert savedParams.roi == Rect(0.1, 0.1, 0.6, 0.6)
        assert savedParams.humanTrackingParams.inactiveTracksLifetime == 4
        assert savedParams.humanTrackingParams.reIDMatchingDetectionsCount == 4
        assert pytest.approx(savedParams.humanTrackingParams.iouConnectionThreshold) == 0.123
        assert pytest.approx(savedParams.humanTrackingParams.reIDMatchingThreshold) == 0.123
        assert pytest.approx(savedParams.humanTrackingParams.removeHorizontalRatio) == 0.123
        assert savedParams.humanTrackingParams.removeOverlappedStrategy == "SCORE"

    assertParams(params)
    te = teFabric()
    streamId = te.registerStream(params)
    params = te.getStreamParams(streamId)
    assertParams(params)
    streamId = te.registerStream(params)
    params = te.getStreamParams(streamId)
    assertParams(params)


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


def test_te_multiframe():
    """Test multiframe per stream traking"""
    te1 = teFabric()
    te2 = teFabric()
    img1 = VLImage.load(filename=ONE_FACE)
    img2 = VLImage.load(filename=CLEAN_ONE_FACE)
    streamId1 = te1.registerStream()
    streamId2 = te1.registerStream()
    streamId3 = te2.registerStream()
    streamId4 = te2.registerStream()

    frame1 = Frame(image=img1, streamId=streamId1, frameNumber=0)
    frame2 = Frame(image=shiftImage(img1, 5, 5), streamId=streamId1, frameNumber=1)

    frame3 = Frame(image=img2, streamId=streamId2, frameNumber=0)
    frame4 = Frame(image=shiftImage(img2, 5, 5), streamId=streamId2, frameNumber=1)

    frame5 = Frame(image=img1, streamId=streamId3, frameNumber=0)
    frame6 = Frame(image=shiftImage(img1, 5, 5), streamId=streamId3, frameNumber=1)

    frame7 = Frame(image=img2, streamId=streamId4, frameNumber=0)
    frame8 = Frame(image=shiftImage(img2, 5, 5), streamId=streamId4, frameNumber=1)

    res1 = te2.track([frame5, frame7])
    res2 = te2.track([frame6, frame8])
    res3 = te1.track([frame1, frame3, frame4, frame2])
    assert res1[0].humanTracks[0].face.bbox.asDict() == res3[0].humanTracks[0].face.bbox.asDict()
    assert res1[1].humanTracks[0].face.bbox.asDict() == res3[2].humanTracks[0].face.bbox.asDict()
    assert res2[0].humanTracks[0].face.bbox.asDict() == res3[1].humanTracks[0].face.bbox.asDict()
    assert res2[1].humanTracks[0].face.bbox.asDict() == res3[3].humanTracks[0].face.bbox.asDict()


def teFabricLandmarks(detectBody=0, detectFace=0, detectLandmarks5=0):
    faceEngine = VLFaceEngine()
    trackEngineProvider = TrackEngineSettingsProvider()
    trackEngineProvider.detectors.useFaceDetector = detectFace
    trackEngineProvider.detectors.useBodyDetector = detectBody
    trackEngineProvider.faceTracking.faceLandmarksDetection = detectLandmarks5
    return VLTrackEngine(faceEngine, trackEngineConf=trackEngineProvider)


@pytest.mark.parametrize("detectFace,detectBody", [(1, 0), (0, 1), (1, 1)])
def test_return_face_landmarks(detectFace, detectBody):
    """Return or not landmarks test"""
    te = teFabricLandmarks(detectFace=detectFace, detectBody=detectBody, detectLandmarks5=1)
    img = VLImage.load(filename=ONE_FACE)
    streamId = te.registerStream()
    frame = Frame(image=img, streamId=streamId, frameNumber=1)

    if detectBody:
        te.track([frame])
        res = te.closeStream(streamId)
        track = res[0].humanTracks[0]
    else:
        res = te.track([frame])
        track = res[0].humanTracks[0]

    if detectBody and detectFace:
        assert track.face.detection.landmarks5 is None
    else:
        assert track.face.detection.landmarks5 is not None


@pytest.mark.parametrize("detectLandmarks", [0, 1])
def test_detect_body_landmarks(detectLandmarks):
    """Detect or not body landmarks test"""

    te = teFabricLandmarks(detectBody=1)
    img = VLImage.load(filename=ONE_FACE)
    streamId = te.registerStream()
    frame = Frame(image=img, streamId=streamId, frameNumber=1)
    te.track([frame])
    res = te.closeStream(streamId)
    track = res[0].humanTracks[0]


@pytest.mark.parametrize("detectLandmarks", [0, 1])
def test_detect_face_landmarks(detectLandmarks):
    """Detect or not face landmarks test"""

    te = teFabricLandmarks(detectFace=1, detectLandmarks5=detectLandmarks)
    img = VLImage.load(filename=ONE_FACE)
    streamId = te.registerStream()
    frame = Frame(image=img, streamId=streamId, frameNumber=1)
    res = te.track([frame])
    track = res[0].humanTracks[0]
    if not detectLandmarks:
        assert track.face.detection.landmarks5 is None
    else:
        assert track.face.detection.landmarks5 is not None
