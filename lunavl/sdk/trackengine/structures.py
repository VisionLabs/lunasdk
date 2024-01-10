from typing import Generic, List, Literal, Optional, TypeVar

from FaceEngine import Face, Human, TrackEngine as te
from FaceEngine.TrackEngine import (
    BodyTrackData,
    FaceTrackData,
    HumanTrackInfo,
    HumanTrackingRemoveOverlappedStrategyType,
    PyTrackingResult,
)

from lunavl.sdk.detectors.bodydetector import BodyDetection, Landmarks17
from lunavl.sdk.detectors.facedetector import FaceDetection, Landmarks5
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage


class Frame:
    """
    Frame container

    Attributes:
        coreFrame: core te Frame
        frameNumber: frame sequence number
        streamId: stream id
        image: image
    """

    def __init__(self, image: VLImage, frameNumber: int, streamId: int):
        frame = te.Frame()
        frame.id = frameNumber
        frame.image = image.coreImage
        self.coreFrame = frame
        self.image = image
        self.streamId = streamId
        self.frameNumber = frameNumber


def _getOptionalValue(attr):
    if attr.isValid():
        return attr.value()
    return None


def _setOptionalValue(attr, value):
    if value is not None:
        attr.set(value)


class HumanTrackingParams:
    """
    Human tracking params
    """

    def __init__(
        self,
        inactiveTracksLifetime: Optional[int] = None,
        iouConnectionThreshold: Optional[float] = None,
        reIDMatchingDetectionsCount: Optional[int] = None,
        reIDMatchingThreshold: Optional[float] = None,
        removeHorizontalRatio: Optional[float] = None,
        removeOverlappedStrategy: Optional[Literal["NONE", "BOTH", "SCORE"]] = None,
        *,
        coreParams: te.HumanTrackingStreamParams = None,
    ):
        """

        Args:
            inactiveTracksLifetime: lifetime of inactive body tracks, which are used for reID. It's measured in
              frames count and used for matching tracks to each other
            iouConnectionThreshold: IOU value threshold, used for matching tracks and detections
            reIDMatchingDetectionsCount: count of detections, that track must have to be matched by reID
            reIDMatchingThreshold: reID value threshold, used for matching tracks to each other
            removeHorizontalRatio: width to height ratio threshold, used for removing horizontal detections
            removeOverlappedStrategy: strategy, used for removing overlapped detections after (re)detect
            coreParams: external core params
        """
        if coreParams:
            self.coreHumanTrackingParamsOpt = te.HumanTrackingStreamParamsOpt()
            self.coreHumanTrackingParams = coreParams
        else:
            self.coreHumanTrackingParamsOpt = te.HumanTrackingStreamParamsOpt()
            self.coreHumanTrackingParams = te.HumanTrackingStreamParams()
        self._setOptionalValue("inactiveTracksLifetime", inactiveTracksLifetime)
        self._setOptionalValue("iouConnectionThreshold", iouConnectionThreshold)
        self._setOptionalValue("reIDMatchingDetectionsCount", reIDMatchingDetectionsCount)
        self._setOptionalValue("reIDMatchingThreshold", reIDMatchingThreshold)
        self._setOptionalValue("removeHorizontalRatio", removeHorizontalRatio)
        if removeOverlappedStrategy:
            ros = getattr(HumanTrackingRemoveOverlappedStrategyType, removeOverlappedStrategy)
            self.coreHumanTrackingParamsOpt.removeOverlappedStrategyOpt.set(ros)
            self.coreHumanTrackingParams.removeOverlappedStrategy = ros
        else:
            ros = self.coreHumanTrackingParams.removeOverlappedStrategy
            self.coreHumanTrackingParamsOpt.removeOverlappedStrategyOpt.set(ros)

    def _setOptionalValue(self, attr, value):
        if value is None:
            value = self.coreHumanTrackingParams.__getattribute__(f"{attr}")
        else:
            self.coreHumanTrackingParams.__setattr__(f"{attr}", value)
        self.coreHumanTrackingParamsOpt.__getattribute__(f"{attr}Opt").set(value)

    @property
    def inactiveTracksLifetime(self) -> int:
        return self.coreHumanTrackingParams.inactiveTracksLifetime

    @inactiveTracksLifetime.setter
    def inactiveTracksLifetime(self, value: int):
        self._setOptionalValue("inactiveTracksLifetime", value)

    @property
    def iouConnectionThreshold(self) -> float:
        return self.coreHumanTrackingParams.iouConnectionThreshold

    @iouConnectionThreshold.setter
    def iouConnectionThreshold(self, value: float):
        self._setOptionalValue("iouConnectionThreshold", value)

    @property
    def reIDMatchingDetectionsCount(self) -> int:
        return self.coreHumanTrackingParams.reIDMatchingDetectionsCount

    @reIDMatchingDetectionsCount.setter
    def reIDMatchingDetectionsCount(self, value: int):
        self._setOptionalValue("reIDMatchingDetectionsCount", value)

    @property
    def reIDMatchingThreshold(self) -> float:
        return self.coreHumanTrackingParams.reIDMatchingThreshold

    @reIDMatchingThreshold.setter
    def reIDMatchingThreshold(self, value: float):
        self._setOptionalValue("reIDMatchingThreshold", value)

    @property
    def removeHorizontalRatio(self) -> float:
        return self.coreHumanTrackingParams.removeHorizontalRatio

    @removeHorizontalRatio.setter
    def removeHorizontalRatio(self, value: float):
        self._setOptionalValue("removeHorizontalRatio", value)

    @property
    def removeOverlappedStrategy(self) -> str:
        return self.coreHumanTrackingParams.removeOverlappedStrategy.name

    @removeOverlappedStrategy.setter
    def removeOverlappedStrategy(self, value: str):
        ros = getattr(HumanTrackingRemoveOverlappedStrategyType, value)
        self._setOptionalValue("removeHorizontalRatio", ros)


class StreamParams:
    """
    General stream params
    """

    def __init__(
        self,
        callbackBufferSize: Optional[int] = None,
        detectorScaling: Optional[bool] = None,
        detectorStep: Optional[int] = None,
        framesBufferSize: Optional[int] = None,
        roi: Optional[Rect] = None,
        humanTrackingParams: Optional[HumanTrackingParams] = None,
        killIntersectedIOUThreshold: Optional[float] = None,
        minimalTrackLength: Optional[int] = None,
        scaledSize: Optional[int] = None,
        skipFrames: Optional[int] = None,
        trackingResultsBufferSize: Optional[int] = None,
        useFrg: Optional[bool] = None,
        *,
        coreParams: te.StreamParams = None,
    ):
        """

        Args:
            callbackBufferSize:  Buffer size for the callbacks. The larger the buffer is, the higher performance is
                ensured, but memory consumption may be higher, [1..]. note: doesn't affect estimator API
            detectorScaling: Scale frame before detection/FRG for performance reasons, [0, 1]
            detectorStep: The count of frames between frames with full detection, [1 .. 30]
            framesBufferSize: ize of the internal storage buffer for all frames (for one stream), [10 ..].
                The larger the buffer is, the higher performance is ensured, but memory consumption may be higher
                note: minimal value is 10, because of internal algorithms requirements, e.g. batching-->
                note: doesn't affect estimator API (callback-mode = 0)
            roi: relative ROI for human tracking
            humanTrackingParams: human tracking params
            killIntersectedIOUThreshold:  value, will be killed
            minimalTrackLength: minimum detections (detect/redetect) count for track (see `TrackInfo::detectionsCount`)
               to return it in tracking results (parameter is ignored for human tracking)
            scaledSize: If scaling is enable, frame will be scaled to this size in pixels for detection step
              (by the max dimension of width/height). Upper scaling is not possible.
            skipFrames: If track wasn't updated by detect/redetect for this number of frames, then track is finished ('36' by default).
               note: very high values may lead to performance degradation. Parameter doesn't affect on human tracking
            trackingResultsBufferSize: buffer size for the stored tracking results
            useFrg: Whether to enable foreground subtractor or not
            coreParams: external core params
        """
        if coreParams:
            self.coreStreamParams = coreParams
            self.coreStreamParamsOpt = te.StreamParamsOpt()
        else:
            self.coreStreamParams = te.StreamParams()
            self.coreStreamParamsOpt = te.StreamParamsOpt()

        self._setOptionalValue("callbackBufferSize", callbackBufferSize)
        self._setOptionalValue("detectorScaling", detectorScaling)
        self._setOptionalValue("detectorStep", detectorStep)
        self._setOptionalValue("framesBufferSize", framesBufferSize)
        if roi:
            self._setOptionalValue("humanRelativeROI", roi.coreRectF)
        else:
            self.coreStreamParamsOpt.humanRelativeROIOpt.set(self.coreStreamParams.humanRelativeROI)

        if humanTrackingParams:
            self.coreStreamParamsOpt.humanTrackingParams = humanTrackingParams.coreHumanTrackingParamsOpt
            self.coreStreamParams.humanTrackingParams = humanTrackingParams.coreHumanTrackingParams
        else:
            humanParams = HumanTrackingParams(coreParams=self.coreStreamParams.humanTrackingParams)
            self.coreStreamParamsOpt.humanTrackingParams = humanParams.coreHumanTrackingParamsOpt

        self._setOptionalValue("killIntersectedIOUThreshold", killIntersectedIOUThreshold)
        self._setOptionalValue("minimalTrackLength", minimalTrackLength)
        self._setOptionalValue("scaledSize", scaledSize)
        self._setOptionalValue("skipFrames", skipFrames)
        self._setOptionalValue("trackingResulsBufferSize", trackingResultsBufferSize)
        self._setOptionalValue("useFrg", useFrg)

    def _setOptionalValue(self, attr, value):
        if value is None:
            value = self.coreStreamParams.__getattribute__(f"{attr}")
        else:
            self.coreStreamParams.__setattr__(f"{attr}", value)
        self.coreStreamParamsOpt.__getattribute__(f"{attr}Opt").set(value)

    @property
    def callbackBufferSize(self) -> int:
        return self.coreStreamParams.callbackBufferSize

    @callbackBufferSize.setter
    def callbackBufferSize(self, value: int):
        self._setOptionalValue("callbackBufferSize", value)

    @property
    def detectorScaling(self) -> bool:
        return self.coreStreamParams.detectorScaling

    @detectorScaling.setter
    def detectorScaling(self, value: bool):
        self._setOptionalValue("detectorScaling", value)

    @property
    def framesBufferSize(self) -> int:
        return self.coreStreamParams.framesBufferSize

    @framesBufferSize.setter
    def framesBufferSize(self, value: int):
        self._setOptionalValue("framesBufferSize", value)

    @property
    def detectorStep(self) -> int:
        return self.coreStreamParams.detectorStep

    @detectorStep.setter
    def detectorStep(self, value: int):
        self._setOptionalValue("detectorStep", value)

    @property
    def roi(self) -> Rect:
        rect = self.coreStreamParams.humanRelativeROI
        return Rect.fromCoreRect(rect)

    @roi.setter
    def roi(self, value: Rect):
        self._setOptionalValue("humanRelativeROI", value.coreRectF)

    @property
    def killIntersectedIOUThreshold(self) -> float:
        return self.coreStreamParams.killIntersectedIOUThreshold

    @killIntersectedIOUThreshold.setter
    def killIntersectedIOUThreshold(self, value: float):
        self._setOptionalValue("killIntersectedIOUThreshold", value)

    @property
    def humanTrackingParams(self) -> HumanTrackingParams:
        return HumanTrackingParams(coreParams=self.coreStreamParams.humanTrackingParams)

    @humanTrackingParams.setter
    def humanTrackingParams(self, value: HumanTrackingParams):
        self.coreStreamParamsOpt.humanTrackingParams = value.coreHumanTrackingParamsOpt
        self.coreStreamParams.humanTrackingParams = value.coreHumanTrackingParams

    @property
    def minimalTrackLength(self) -> int:
        return self.coreStreamParams.minimalTrackLength

    @minimalTrackLength.setter
    def minimalTrackLength(self, value: int):
        self._setOptionalValue("minimalTrackLength", value)

    @property
    def scaledSize(self) -> int:
        return self.coreStreamParams.scaledSize

    @scaledSize.setter
    def scaledSize(self, value: int):
        self._setOptionalValue("scaledSize", value)

    @property
    def skipFrames(self) -> int:
        return self.coreStreamParams.skipFrames

    @skipFrames.setter
    def skipFrames(self, value: int):
        self._setOptionalValue("skipFrames", value)

    @property
    def trackingResultsBufferSize(self) -> int:
        return self.coreStreamParams.trackingResulsBufferSize

    @trackingResultsBufferSize.setter
    def trackingResultsBufferSize(self, value: int):
        self._setOptionalValue("trackingResulsBufferSize", value)

    @property
    def useFrg(self) -> int:
        return self.coreStreamParams.useFrg

    @useFrg.setter
    def useFrg(self, value: int):
        self._setOptionalValue("useFrg", value)


TrackedObject = TypeVar("TrackedObject")
TrackedDetectionObject = TypeVar("TrackedDetectionObject")


class BaseTrackObject(Generic[TrackedObject, TrackedDetectionObject]):
    """Base class for detected on frame object"""

    def __init__(self, coreEstimation: TrackedObject, image: VLImage):
        self.coreEstimation = coreEstimation
        self.image = image

    @property
    def bbox(self) -> Rect:
        """Object bbox. It is a detector or tracker work result"""
        return Rect.fromCoreRect(self.coreEstimation.detection.getRawRect())  # type: ignore

    @property
    def lastDetectionFrame(self) -> int:
        """Last frame with detection"""
        return self.coreEstimation.lastDetectionFrameId  # type: ignore

    @property
    def firstFrame(self) -> int:
        """First frame with detection"""
        return self.coreEstimation.firstFrameId  # type: ignore

    @property
    def detection(self) -> TrackedObject | None:
        raise NotImplemented

    def asDict(self) -> dict:
        return {
            "bbox": self.bbox.asDict(),
            "first_frame": self.firstFrame,
            "last_detection_frame": self.lastDetectionFrame,
            "detection": self.detection.asDict() if self.detection else None,  # type: ignore
        }

    def __repr__(self) -> str:
        return str(self.asDict())


class FaceTrack(BaseTrackObject[FaceTrackData, FaceDetection]):
    """
    Containers for track body detection
    """

    __slots__ = ["_detection"]

    def __init__(self, coreEstimation, image):
        super().__init__(coreEstimation, image)
        self._detection: FaceDetection | None = None

    @property
    def detection(self) -> Optional[FaceDetection]:
        """Get honest face detection. This detection is a result of detector work (not tracker)"""
        if self._detection is None:
            if not self.coreEstimation.isDetector:
                return None
            coreFace = Face(self.image.coreImage, self.coreEstimation.detection)
            face = FaceDetection(coreFace, self.image)
            coreLandmarks = self.coreEstimation.landmarks
            if coreLandmarks and not all((landmark.x == 0 and landmark.y == 0 for landmark in coreLandmarks)):
                coreFace.landmarks5_opt.set(coreLandmarks)
                face.landmarks5 = Landmarks5(coreLandmarks)
            self._detection = face
        return self._detection


class BodyTrack(BaseTrackObject[BodyTrackData, BodyDetection]):
    """
    Containers for track body detection
    """

    __slots__ = ["_detection"]

    def __init__(self, coreEstimation, image):
        super().__init__(coreEstimation, image)
        self._detection: BodyDetection | None = None

    @property
    def detection(self) -> Optional[BodyDetection]:
        """Get honest body detection. This detection is a result of detector work (not tracker)"""
        if self._detection is None:
            if not self.coreEstimation.isDetector:
                return None
            coreBody = Human()
            coreBody.img = self.image.coreImage
            coreBody.detection = self.coreEstimation.detection
            self._detection = BodyDetection(coreBody, self.image)
            coreLandmarks = self.coreEstimation.landmarks
            if coreLandmarks and not all(
                (landmark.point.x == 0 and landmark.point.y == 0 for landmark in coreLandmarks)
            ):
                coreBody.landmarks17_opt.set(coreLandmarks)
                self._detection.landmarks17 = Landmarks17(coreLandmarks)
        return self._detection


class HumanTrack:
    """
    Human track estimation on a frame

    Attributes:

    """

    def __init__(self, coreEstimation: HumanTrackInfo, image: VLImage):
        self.coreEstimation = coreEstimation
        self.image = image

    @property
    def face(self) -> Optional[FaceTrack]:
        """Face detection"""
        if not self.coreEstimation.faceOpt.isValid():
            return None
        face = self.coreEstimation.faceOpt.value()
        if face:
            return FaceTrack(face, self.image)
        return None

    @property
    def body(self) -> Optional[BodyTrack]:
        """Body detection"""
        if not self.coreEstimation.bodyOpt.isValid():
            return None
        body = self.coreEstimation.bodyOpt.value()
        if body:
            return BodyTrack(body, self.image)
        return None

    @property
    def trackId(self) -> int:
        """Track id"""
        return self.coreEstimation.trackId

    def asDict(self) -> dict:
        return {
            "track_id": self.coreEstimation.trackId,
            "face": self.face.asDict() if self.face else None,
            "body": self.body.asDict() if self.body else None,
        }

    def __repr__(self) -> str:
        return str(self.asDict())


class TrackingResult:
    """
    Tracking result by a frame

    Attributes:
        _humanTracks: cached property
        coreResult: core tracks
    """

    def __init__(self, coreResult: PyTrackingResult):
        self.coreResult = coreResult
        self._humanTracks: Optional[List[HumanTrack]] = None

    @property
    def image(self) -> VLImage:
        """Get image origin"""
        return VLImage(self.coreResult.image)

    @property
    def streamId(self) -> int:
        """Get stream id"""
        return self.coreResult.streamId

    @property
    def frameId(self):
        """Get frame id"""
        return self.coreResult.frameId

    @property
    def humanTracks(self) -> List[HumanTrack]:
        """Get all human tracks"""
        if self._humanTracks is None:
            self._humanTracks = [HumanTrack(humanTrack, self.image) for humanTrack in self.coreResult.humanTracks]
        return self._humanTracks

    @property
    def startTracks(self):
        """Get a started trak id list"""
        return [data.trackId for data in self.coreResult.trackStart]

    @property
    def endTracks(self):
        """Get a ended trak id list"""
        return [data.trackId for data in self.coreResult.trackEnd]

    def asDict(self) -> dict:
        return {
            "human_tracks": [ht.asDict() for ht in self.humanTracks],
            "end_tracks": self.startTracks,
            "start_tracks": self.startTracks,
        }

    def __repr__(self) -> str:
        return str(self.asDict())
