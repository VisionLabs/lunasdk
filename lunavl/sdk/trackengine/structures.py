from typing import Optional, Union

from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from FaceEngine import TrackEngine as te


class Frame:
    def __init__(self, image: VLImage, frameNumber: int, streamId: int):
        frame = te.Frame()
        frame.id = frameNumber
        frame.image = image.coreImage
        self.coreFrame = frame
        self.image = image
        self.streamId = streamId


class HumanTrackingParams:
    def __init__(self,
                 inactiveTracksLifetime: Optional[int] = None,
                 IOUConnectionThreshold: Optional[float] = None,
                 reIDMatchingDetectionsCount: Optional[int] = None,
                 reIDMatchingThreshold: Optional[float] = None,
                 removeHorizontalRatio: Optional[float] = None,
                 removeOverlappedStrategy: Optional[str] = None,
                 *,
                 coreParams: Union[te.HumanTrackingStreamParamsOpt, te.HumanTrackingStreamParams] = None
                 ):
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
    def __init__(self,
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
                 coreParams: Union[te.StreamParamsOpt, te.StreamParams] = None
                 ):
        if coreParams:
            self.coreStreamParamsOpt = coreParams
        else:
            self.coreStreamParamsOpt = te.StreamParamsOpt()
        self.coreStreamParamsOpt.callback_buffer_size_opt = callbackBufferSize
        self.coreStreamParamsOpt.detector_scaling_opt = detectorScaling
        self.coreStreamParamsOpt.detector_step_opt = detectorStep
        self.coreStreamParamsOpt.frames_buffer_size_opt = framesBufferSize
        self.coreStreamParamsOpt.human_relative_ROI_opt = roi
        if humanTrackingParams:
            self.coreStreamParamsOpt.human_tracking_params = humanTrackingParams.coreHumanTrackingParams
        self.coreStreamParamsOpt.kill_intersected_IOU_threshold_opt = killIntersectedIOUThreshold
        self.coreStreamParamsOpt.minimal_track_length_opt = minimalTrackLength
        self.coreStreamParamsOpt.scaled_size_opt = scaledSize
        self.coreStreamParamsOpt.skip_frames_opt = skipFrames
        self.coreStreamParamsOpt.tracking_resuls_buffer_size_opt = trackingResultsBufferSize
        self.coreStreamParamsOpt.use_frg_opt = useFrg
        self.getter = lambda name

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
