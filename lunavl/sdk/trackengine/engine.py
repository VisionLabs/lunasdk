from typing import Dict, List, Literal, Optional, TypeVar, Union, overload

import TrackEngine as te

from ..async_task import AsyncTask, DefaultPostprocessingFactory
from ..faceengine.engine import VLFaceEngine
from ..launch_options import DeviceClass, LaunchOptions
from .setting_provider import TrackEngineSettingsProvider
from .structures import Frame, StreamParams, TrackingResult

TrackedObject = TypeVar("TrackedObject")

POST_PROCESSING = DefaultPostprocessingFactory(TrackingResult)


class VLTrackEngine:
    """
    Wraper on TrackEngine.

    Attributes:
        _faceEngine (PyITrackEngine): python C++ binding on IFaceEngine, Root LUNA SDK object interface
        _streamIdToStream: map stream id to core streams
        _trackEngine: core te instance
    """

    def __init__(
        self,
        faceEngine: VLFaceEngine,
        trackEngineConf: Optional[Union[str, TrackEngineSettingsProvider]] = None,
        launchOptions: Optional[LaunchOptions] = None,
    ):
        """

        Args:
            faceEngine: faceEngine instance
            trackEngineConf: trackEngine conf

        """
        launchOptions = faceEngine.getLaunchOptions(launchOptions)
        if trackEngineConf is None:
            self.trackEngineProvider = TrackEngineSettingsProvider()
        elif isinstance(trackEngineConf, str):
            self.trackEngineProvider = TrackEngineSettingsProvider(trackEngineConf)
        else:
            self.trackEngineProvider = trackEngineConf
        self.trackEngineProvider.other.callbackMode = 0
        self._faceEngine = faceEngine
        self._trackEngine = te.createTrackEngineBySettingsProvider(
            faceEngine._faceEngine, self.trackEngineProvider.coreProvider, launchOptions=launchOptions.coreLaunchOptions
        )

        self._streamIdToStream: Dict[int, te.PyIStream] = {}

    @overload  # type: ignore
    def track(self, frames: list[Frame], asyncEstimate: Literal[False] = False) -> List[TrackingResult]: ...

    @overload
    def track(self, frames: list[Frame], asyncEstimate: Literal[True]) -> AsyncTask[List[TrackingResult]]: ...

    def track(self, frames: list[Frame], asyncEstimate: bool = False):
        """
        Updates stream tracks by new frame per each stream and returns ready tracking
        results data for passed streams.Function returns only ready tracking results per each stream, so it can return
        tracking results for Stream previously passed frames as well as not return results for current passed frame.

        Warnings:
            This function is not thread safe. User must run only one coroutine simultaneously.

        Args:
            frames: or each stream should be no more than one frame. list must contain no more one frame
            for each stream.
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated tracks for processed frames(may differ from input frames) if asyncEstimate is false otherwise
            async task.
        """
        streamIdsToFrames: Dict[int, List[te.Frame]] = {}
        streamIds = []
        coreStreamFrames = []
        for frame in frames:
            streamIdsToFrames.setdefault(frame.streamId, []).append(frame.coreFrame)
        for streamId, coreFrames in streamIdsToFrames.items():
            coreStreamFrames.append(coreFrames)
            streamIds.append(streamId)

        if not asyncEstimate:
            res = self._trackEngine.trackMultiFrame(streamIds, coreStreamFrames)
            return POST_PROCESSING.postProcessingBatch(*res)
        task = self._trackEngine.async_trackMultiFrame(streamIds, coreStreamFrames)
        return AsyncTask(task, POST_PROCESSING.postProcessingBatch)

    def registerStream(self, params: Optional[StreamParams] = None) -> int:
        """
        Register stream.
        Args:
            params: stream processing params

        Returns:
            stream id
        """
        if params:
            coreStream = self._trackEngine.createStreamWithParams(params.coreStreamParamsOpt)
        else:
            coreStream = self._trackEngine.createStream()
        self._streamIdToStream[coreStream.getId()] = coreStream
        return coreStream.getId()

    def closeStream(self, streamId: int) -> list[TrackingResult]:
        """
        Finishes all current tracks and returns all remaining tracking results. If `reset` is true, then resets Stream's
        state to initial. Stream can be used for tracking further after call of this func. It can't be called during
        Stream processing, otherwise UB, so users should assure, that func is called when Stream doesn't have any frame
        for processing.

        Args:
            streamId: stream id

        Returns:
            estimated tracks
        """
        stream = self._streamIdToStream.pop(streamId)
        res = stream.stop(True)
        tarcks = [TrackingResult(r) for r in res if r.detections]
        return tarcks

    def getStreamParams(self, streamId: int) -> StreamParams:
        """
        Get streams params

        Args:
            streamId: stream id
        Returns:
            stream params
        """
        coreStream = self._streamIdToStream[streamId]
        params = StreamParams(coreParams=coreStream.getParams())
        return params

    def reconfigure(self, streamId: int, params: StreamParams):
        coreStream = self._streamIdToStream[streamId]
        coreStream.reconfigure(params.coreStreamParamsOpt)
