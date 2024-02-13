"""
An mouth state estimation example
"""

import asyncio
import pprint

from resources import EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.trackengine.engine import VLTrackEngine
from lunavl.sdk.trackengine.setting_provider import TrackEngineSettingsProvider
from lunavl.sdk.trackengine.structures import Frame


def estimateHumanTrack(detectFace=1, detectBody=1):
    """
    Estimate human track
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    trackEngineProvider = TrackEngineSettingsProvider()
    trackEngineProvider.detectors.useFaceDetector = detectFace
    trackEngineProvider.detectors.useBodyDetector = detectBody
    trackengine = VLTrackEngine(faceEngine, trackEngineConf=trackEngineProvider)
    streamId = trackengine.registerStream()
    res = trackengine.track([Frame(image, 1, streamId)])
    pprint.pprint(res)
    res = trackengine.track([Frame(image, 2, streamId)])
    pprint.pprint(res)
    res = trackengine.closeStream(streamId)
    pprint.pprint(res)


async def asyncEstimateHumanTrack():
    """
    Async estimate human track
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    trackEngineProvider = TrackEngineSettingsProvider()
    trackEngineProvider.detectors.useFaceDetector = 1
    trackEngineProvider.detectors.useBodyDetector = 1
    trackengine = VLTrackEngine(faceEngine, trackEngineConf=trackEngineProvider)
    streamId = trackengine.registerStream()
    res = await trackengine.track([Frame(image, 1, streamId)], asyncEstimate=True)
    pprint.pprint(res)
    res = trackengine.track([Frame(image, 2, streamId)], asyncEstimate=True).get()
    pprint.pprint(res)
    res = trackengine.closeStream(streamId)
    pprint.pprint(res)


if __name__ == "__main__":
    estimateHumanTrack(0, 1)
    estimateHumanTrack(1, 0)
    estimateHumanTrack(1, 1)
    asyncio.run(asyncEstimateHumanTrack())
