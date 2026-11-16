LUNA SDK
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:


   sdk/modules

Getting Started
---------------

Attributes estimation example:

>>> from lunavl.sdk.faceengine.engine import VLFaceEngine
>>> from lunavl.sdk.luna_faces import VLFaceDetector
>>> from lunavl.sdk.luna_faces import VLImage
>>> VLFaceDetector.initialize(VLFaceEngine())
>>> detector = VLFaceDetector()
>>> image = VLImage.load(url='https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg')
>>> detection = detector.detectOne(image)
>>> detection.basicAttributes.age
16.0
>>> detection.emotions.predominateEmotion.name
'Happiness'
