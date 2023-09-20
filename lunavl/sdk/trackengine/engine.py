import os
from pathlib import Path
from typing import Optional, Union, Dict
import TrackEngine as te

from .structures import Frame, StreamParams
from ..faceengine.engine import VLFaceEngine
from ..faceengine.setting_provider import FaceEngineSettingsProvider, BaseSettingsProvider


class TrackEngineSettingsProvider(BaseSettingsProvider):
    # default configuration filename.
    defaultConfName = "trackengine.conf"


class VLTrackEngine:
    """
    Wraper on FaceEngine.

    Attributes:
        dataPath (str): path to a faceengine data folder
        faceEngineProvider (FaceEngineSettingsProvider): face engine settings provider
        runtimeProvider (RuntimeSettingsProvider): runtime settings provider
        _faceEngine (PyIFaceEngine): python C++ binding on IFaceEngine, Root LUNA SDK object interface
    """

    def __init__(
            self,
            faceEngine: VLFaceEngine,
            trackEngineConf: Optional[Union[str, TrackEngineSettingsProvider]] = None,
    ):
        if trackEngineConf is None:
            self.trackEngineProvider = TrackEngineSettingsProvider()
        elif isinstance(trackEngineConf, str):
            self.trackEngineProvider = TrackEngineSettingsProvider(trackEngineConf)
        else:
            self.trackEngineProvider = trackEngineConf
        self._trackEngine = te.createTrackEngine(faceEngine._faceEngine,
                                                 f"{self.trackEngineProvider.pathToConfig.parent}/trackengine.conf")
        self._streamIdToStream: Dict[int, te.PyIStream] = {}

    def track(self, frames: list[Frame]):
        streamIds = [frame.streamId for frame in frames]
        frames = [frame.coreFrame for frame in frames]
        res = self._trackEngine.track(streamIds, frames)
        return res

    def registerStream(self, params: Optional[StreamParams] = None) -> int:
        if params:
            coreStream = self._trackEngine.createStreamWithParams(params.coreStreamParamsOpt)
        else:
            coreStream = self._trackEngine.createStream()
        self._streamIdToStream[coreStream.getId()] = coreStream
        return coreStream.getId()

    def closeStream(self, streamId: int):
        stream = self._streamIdToStream.pop(streamId)
        res = stream.coreStream.reinit(True)
        stream.coreStream = None
        return res

    def getStreamParams(self, streamId: int) ->:
        coreStream = self._streamIdToStream[streamId]
        params = coreStream.getParams()

