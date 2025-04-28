"""
Module realize simple examples following features:
    * image modification estimation
"""

import asyncio
import pprint

from resources import EXAMPLE_1, EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage


def estimateImageModification():
    """
    Example of a image modification estimation.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    imageModificationEstimator = faceEngine.createImageModificationEstimator()
    #: estimate
    pprint.pprint(imageModificationEstimator.estimate(image))

    image2 = VLImage.load(filename=EXAMPLE_1)
    #: estimate batch
    pprint.pprint(imageModificationEstimator.estimateBatch([image, image2]))


async def asyncEstimateImageModificationTrack():
    """
    Async estimate human track
    """
    faceEngine = VLFaceEngine()
    modificationEstimator = faceEngine.createImageModificationEstimator()
    image = VLImage.load(filename=EXAMPLE_O)
    modification = await modificationEstimator.estimate(image, asyncEstimate=True)
    pprint.pprint(modification)
    task1 = modificationEstimator.estimate(image, asyncEstimate=True)
    task2 = modificationEstimator.estimate(image, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get())


if __name__ == "__main__":
    estimateImageModification()
    asyncio.run(asyncEstimateImageModificationTrack())
