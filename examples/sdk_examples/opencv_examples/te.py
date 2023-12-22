"""
Human tracking visualization example.
"""
import asyncio
from asyncio import Queue

import cv2  # pylint: disable=E0611,E0401
from numpy import ndarray

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.trackengine.engine import VLTrackEngine
from lunavl.sdk.trackengine.structures import Frame, TrackingResult


def drawTrackBoxes(frame: ndarray, tracks: TrackingResult):
    """Draw bboxes on frame"""

    def drawTrack(bbox):
        cv2.rectangle(
            frame,
            (int(bbox.x), int(face.bbox.y)),
            (int(bbox.x) + int(face.bbox.width), int(face.bbox.y) + int(face.bbox.height)),
            (25, 240, 25),
            2,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(track.trackId), (int(face.bbox.x), int(face.bbox.y) - 10), font, 1, (25, 240, 25), 2)

    for track in tracks.humanTracks:
        if face := track.face:
            drawTrack(face.bbox)
        if body := track.body:
            drawTrack(body.bbox)


async def viewTracking(queue: Queue):
    """Visualize tracks"""
    while True:
        processedFrame: TrackingResult | None = await queue.get()
        if processedFrame is None:
            return
        img = processedFrame.image.asNPArray()
        drawTrackBoxes(img, processedFrame)
        cv2.imshow("Original image", img)
        cv2.waitKey(1)


async def processVideo(videoFile: str, queue: Queue):
    """
    Decode video and send frames to tracking
    """
    faceEngine = VLFaceEngine()
    trackEngine = VLTrackEngine(faceEngine)
    cap = cv2.VideoCapture(videoFile)
    streamId = trackEngine.registerStream()
    frameNumber = 1
    while True:
        # Capture frame-by-frame
        ret, cvFrame = await asyncio.to_thread(cap.read)
        if not ret:
            break
        image = VLImage(cvFrame)
        frame = Frame(image, frameNumber, streamId=streamId)
        trackingResults = await trackEngine.track([frame], asyncEstimate=True)
        if trackingResults:
            for resultOnFrame in trackingResults:
                queue.put_nowait(resultOnFrame)

        frameNumber += 1
    trackingResults = trackEngine.closeStream(streamId=streamId)
    if trackingResults:
        for resultOnFrame in trackingResults:
            queue.put_nowait(resultOnFrame)
    queue.put_nowait(None)


async def main(videoFile: str):
    """Track humans on video"""
    visualizeQueue = Queue()
    visualize = asyncio.create_task(viewTracking(visualizeQueue))
    await processVideo(videoFile, visualizeQueue)
    await visualize


if __name__ == "__main__":
    video = "PATH/TO/FILE"
    asyncio.run(main(video))
