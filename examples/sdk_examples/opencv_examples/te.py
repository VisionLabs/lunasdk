"""
Warps visualization example.
"""
import pprint

import cv2  # pylint: disable=E0611,E0401

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.trackengine.engine import VLTrackEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage

import FaceEngine as fe

from lunavl.sdk.trackengine.structures import Frame


def draw_bounding_boxes(frame_to_draw, bboxes_to_draw):
    for t, b in bboxes_to_draw.items():
        cv2.rectangle(frame_to_draw,
                      (b.x, b.y),
                      (b.x + b.width, b.y + b.height),
                      (25, 240, 25),
                      2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_to_draw, str(t), (b.x, b.y - 10), font, 1, (25, 240, 25), 2)

def createWarp():
    """
    Create warp from detection.

    """
    image = VLImage.load(url="https://img.championat.com/s/1350x900/news/big/g/v/oficialno-hbo-snimet-serial-garri-potter_16813235311040883005.jpg")
    faceEngine = VLFaceEngine()
    trackEngine = VLTrackEngine(faceEngine)
    process = True
    cap = cv2.VideoCapture("D:/Абхазия-лазаревское/C0197.MP4")
    frames = {}
    streamId = trackEngine.registerStream()
    x = 0
    while process:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame!")
            process = False
            break

        image = VLImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame = Frame(image, x, streamId=streamId)
        frames = {x: frame}


        res = trackEngine.track([frame])
        x +=1
        if res[1]:
            print(res[1][0].detections, x)
    trackEngine.closeStream(streamId=streamId)




if __name__ == "__main__":
    createWarp()
