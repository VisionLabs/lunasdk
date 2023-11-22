from typing import Generic, List, Literal, Optional, TypeVar, Union

from FaceEngine import Face, Human, Image, TrackEngine as te
from FaceEngine.TrackEngine import BodyTrackData, FaceTrackData, HumanTrackInfo, PyTrackingResult

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


class HumanTrackingParams:
    """
    Human tracking params
    """

    def __init__(
        self,
        inactiveTracksLifetime: Optional[int] = None,
        IOUConnectionThreshold: Optional[float] = None,
        reIDMatchingDetectionsCount: Optional[int] = None,
        reIDMatchingThreshold: Optional[float] = None,
        removeHorizontalRatio: Optional[float] = None,
        removeOverlappedStrategy: Optional[Literal["none", "both", "score"]] = None,
        *,
        coreParams: Union[te.HumanTrackingStreamParamsOpt, te.HumanTrackingStreamParams] = None,
    ):
        """

        Args:
            inactiveTracksLifetime: lifetime of inactive body tracks, which are used for reID. It's measured in
              frames count and used for matching tracks to each other
            IOUConnectionThreshold: IOU value threshold, used for matching tracks and detections
            reIDMatchingDetectionsCount: ount of detections, that track must have to be matched by reID
            reIDMatchingThreshold: reID value threshold, used for matching tracks to each other
            removeHorizontalRatio: width to height ratio threshold, used for removing horizontal detections
            removeOverlappedStrategy: strategy, used for removing overlapped detections after (re)detect
            coreParams: external core params
        """
        if coreParams:
            self.coreHumanTrackingParams = coreParams
        else:
            self.coreHumanTrackingParams = te.HumanTrackingStreamParamsOpt()
        self.coreHumanTrackingParams.inactive_tracks_lifetime_opt = inactiveTracksLifetime
        self.coreHumanTrackingParams.IOU_connection_threshold_opt = IOUConnectionThreshold
        self.coreHumanTrackingParams.reID_matching_detections_count_opt = reIDMatchingDetectionsCount
        self.coreHumanTrackingParams.reID_matching_threshold_opt = reIDMatchingThreshold
        self.coreHumanTrackingParams.remove_horizontal_ratio_opt = removeHorizontalRatio
        self.coreHumanTrackingParams.remove_overlapped_strategy_opt = removeOverlappedStrategy

    @property
    def inactiveTracksLifetime(self) -> int:
        return self.coreHumanTrackingParams.inactive_tracks_lifetime_opt

    @inactiveTracksLifetime.setter
    def inactiveTracksLifetime(self, value: int):
        self.coreHumanTrackingParams.inactive_tracks_lifetime_opt = value

    @property
    def IOUConnectionThreshold(self) -> float:
        return self.coreHumanTrackingParams.IOU_connection_threshold_opt

    @IOUConnectionThreshold.setter
    def IOUConnectionThreshold(self, value: float):
        self.coreHumanTrackingParams.IOU_connection_threshold_opt = value

    @property
    def reIDMatchingDetectionsCount(self):
        return self.coreHumanTrackingParams.reID_matching_detections_count_opt

    @reIDMatchingDetectionsCount.setter
    def reIDMatchingDetectionsCount(self, value: int):
        self.coreHumanTrackingParams.reID_matching_detections_count_opt = value

    @property
    def reIDMatchingThreshold(self) -> float:
        return self.coreHumanTrackingParams.reID_matching_threshold_opt

    @reIDMatchingThreshold.setter
    def reIDMatchingThreshold(self, value: float):
        self.coreHumanTrackingParams.reID_matching_threshold_opt = value

    @property
    def removeHorizontalRatio(self) -> float:
        return self.coreHumanTrackingParams.remove_horizontal_ratio_opt

    @removeHorizontalRatio.setter
    def removeHorizontalRatio(self, value: float):
        self.coreHumanTrackingParams.remove_horizontal_ratio_opt = value

    @property
    def removeOverlappedStrategy(self) -> str:
        return self.coreHumanTrackingParams.remove_overlapped_strategy_opt

    @removeOverlappedStrategy.setter
    def removeOverlappedStrategy(self, value: str):
        self.coreHumanTrackingParams.remove_overlapped_strategy_opt = value


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
        coreParams: Union[te.StreamParamsOpt, te.StreamParams] = None,
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
            self.coreStreamParamsOpt = coreParams
        else:
            self.coreStreamParamsOpt = te.StreamParamsOpt()
        self.coreStreamParamsOpt.callback_buffer_size_opt = callbackBufferSize
        self.coreStreamParamsOpt.detector_scaling_opt = detectorScaling
        self.coreStreamParamsOpt.detector_step_opt = detectorStep
        self.coreStreamParamsOpt.frames_buffer_size_opt = framesBufferSize
        if roi:
            self.coreStreamParamsOpt.human_relative_ROI_opt = roi.coreRectF
        if humanTrackingParams:
            self.coreStreamParamsOpt.human_tracking_params = humanTrackingParams.coreHumanTrackingParams
        self.coreStreamParamsOpt.kill_intersected_IOU_threshold_opt = killIntersectedIOUThreshold
        self.coreStreamParamsOpt.minimal_track_length_opt = minimalTrackLength
        self.coreStreamParamsOpt.scaled_size_opt = scaledSize
        self.coreStreamParamsOpt.skip_frames_opt = skipFrames
        self.coreStreamParamsOpt.tracking_resuls_buffer_size_opt = trackingResultsBufferSize
        self.coreStreamParamsOpt.use_frg_opt = useFrg

    @property
    def callbackBufferSize(self) -> int:
        return self.coreStreamParamsOpt.callback_buffer_size_opt

    @callbackBufferSize.setter
    def callbackBufferSize(self, value: int):
        self.coreStreamParamsOpt.callback_buffer_size_opt = value

    @property
    def detectorScaling(self) -> bool:
        return self.coreStreamParamsOpt.callback_buffer_size_opt

    @detectorScaling.setter
    def detectorScaling(self, value: bool):
        self.coreStreamParamsOpt.detector_scaling_opt = value

    @property
    def framesBufferSize(self) -> int:
        return self.coreStreamParamsOpt.frames_buffer_size_opt

    @framesBufferSize.setter
    def framesBufferSize(self, value: int):
        self.coreStreamParamsOpt.frames_buffer_size_opt = value

    @property
    def detectorStep(self) -> int:
        return self.coreStreamParamsOpt.detector_step_opt

    @detectorStep.setter
    def detectorStep(self, value: int):
        self.coreStreamParamsOpt.detector_step_opt = value

    @property
    def roi(self) -> Rect:
        return Rect.fromCoreRect(self.coreStreamParamsOpt.human_relative_ROI_opt)

    @roi.setter
    def roi(self, value: Rect):
        self.coreStreamParamsOpt.human_relative_ROI_opt = value.coreRectF

    @property
    def killIntersectedIOUThreshold(self) -> float:
        return self.coreStreamParamsOpt.kill_intersected_IOU_threshold_opt

    @killIntersectedIOUThreshold.setter
    def killIntersectedIOUThreshold(self, value: float):
        self.coreStreamParamsOpt.kill_intersected_IOU_threshold_opt = value

    @property
    def humanTrackingParams(self) -> HumanTrackingParams:
        return HumanTrackingParams(coreParams=self.coreStreamParamsOpt.human_tracking_params)

    @humanTrackingParams.setter
    def humanTrackingParams(self, value: HumanTrackingParams):
        self.coreStreamParamsOpt.human_relative_ROI_opt = value.coreHumanTrackingParams

    @property
    def minimalTrackLength(self) -> int:
        return self.coreStreamParamsOpt.minimal_track_length_opt

    @minimalTrackLength.setter
    def minimalTrackLength(self, value: int):
        self.coreStreamParamsOpt.minimal_track_length_opt = value

    @property
    def scaledSize(self) -> int:
        return self.coreStreamParamsOpt.scaled_size_opt

    @scaledSize.setter
    def scaledSize(self, value: int):
        self.coreStreamParamsOpt.scaled_size_opt = value

    @property
    def skipFrames(self) -> int:
        return self.coreStreamParamsOpt.skip_frames_opt

    @skipFrames.setter
    def skipFrames(self, value: int):
        self.coreStreamParamsOpt.skip_frames_opt = value

    @property
    def trackingResultsBufferSize(self) -> int:
        return self.coreStreamParamsOpt.tracking_resuls_buffer_size_opt

    @trackingResultsBufferSize.setter
    def trackingResultsBufferSize(self, value: int):
        self.coreStreamParamsOpt.tracking_resuls_buffer_size_opt = value

    @property
    def useFrg(self) -> int:
        return self.coreStreamParamsOpt.use_frg_opt

    @useFrg.setter
    def useFrg(self, value: int):
        self.coreStreamParamsOpt.use_frg_opt = value


TrackedObject = TypeVar("TrackedObject")
TrackedDetectionObject = TypeVar("TrackedDetectionObject")


class BaseTrackObject(Generic[TrackedObject, TrackedDetectionObject]):
    """Base class for detected on frame object"""

    def __init__(self, coreEstimation: TrackedObject, image: VLImage):
        self.coreEstimation = coreEstimation
        self.image = image

    @property
    def bbox(self):
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

    @property
    def detection(self) -> Optional[FaceDetection]:
        """Get honest face detection. This detection is a result of detector work (not tracker)"""
        if not self.coreEstimation.isDetector:
            return None
        face = FaceDetection(Face(self.image.coreImage, self.coreEstimation.detection), self.image)
        coreLandmarks = self.coreEstimation.landmarks
        if coreLandmarks:
            face.landmarks5 = Landmarks5(coreLandmarks)
        return face


class BodyTrack(BaseTrackObject[BodyTrackData, BodyDetection]):
    """
    Containers for track body detection
    """

    @property
    def detection(self) -> Optional[BodyDetection]:
        """Get honest body detection. This detection is a result of detector work (not tracker)"""
        if not self.coreEstimation.isDetector:
            return None
        coreBody = Human()
        coreBody.img = self.image.coreImage
        coreBody.detection = self.coreEstimation.detection
        body = BodyDetection(coreBody, self.image)
        coreLandmarks = self.coreEstimation.landmarks
        if coreLandmarks:
            body.landmarks17 = Landmarks17(coreLandmarks)
        return body


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
