"""
Module realize transfer functions from/to ML framework tensor to/from VLImage without copy
"""

import contextlib
from typing import Iterator

from FaceEngine import (  # pylint: disable=E0611,E0401
    Image as CoreImage,
)

from .image import VLImage

try:
    from torch import Tensor as PytorchTensor
    from torch.utils.dlpack import from_dlpack, to_dlpack
except ImportError:
    PytorchTensor = to_dlpack = from_dlpack = None  # type: ignore

try:
    from onnxruntime.capi._pybind_state import OrtValue  # type: ignore
except ImportError:
    OrtValue = None  # type: ignore


try:
    from tensorflow import Tensor as TFTensor
    from tensorflow.experimental.dlpack import from_dlpack as tf_from_dlpack, to_dlpack as tf_to_dlpack
except ImportError:
    tf_to_dlpack = tf_from_dlpack = None  # type: ignore


@contextlib.contextmanager
def toPytorch(image: VLImage) -> Iterator[PytorchTensor]:
    """
    Transfer VLImage to Pytorch Tensor without copy.

    Warning:
        use tensor in contextmanager lifecycle scope
    """
    if from_dlpack is None:
        raise RuntimeError("pytorch is not installed, minimal required version see in FSDK")
    coreImage = image.coreImage
    torchTensor = from_dlpack(coreImage)
    yield torchTensor


@contextlib.contextmanager
def fromPytorch(torchTensor: PytorchTensor) -> Iterator[VLImage]:
    """
    Transfer Pytorch Tensor to VLImage without copy.

    Warning:
        use image in contextmanager lifecycle scope
    """
    if to_dlpack is None:
        raise RuntimeError("pytorch is not installed, minimal required version see in FSDK")
    coreImage = CoreImage.from_dlpack(to_dlpack(torchTensor))
    yield VLImage(coreImage)


@contextlib.contextmanager
def toOnnx(image: VLImage) -> Iterator[OrtValue]:
    """
    Transfer VLImage to ONNX without copy.

    Warning:
        use tensor in contextmanager lifecycle scope
    """
    if OrtValue is None:
        raise RuntimeError("onnxruntime is not installed, minimal required version see in FSDK")
    coreImage = image.coreImage
    onnxTensor = OrtValue.from_dlpack(coreImage.__dlpack__(), False)
    yield onnxTensor


@contextlib.contextmanager
def fromOnnx(onnxTensor: OrtValue) -> Iterator[VLImage]:
    """
    Transfer ONNX to VLImage without copy.

    Warning:
        use image in contextmanager lifecycle scope
    """
    coreImage = CoreImage.from_dlpack(onnxTensor.to_dlpack())
    yield VLImage(coreImage)


@contextlib.contextmanager
def toTensorFlow(image: VLImage) -> Iterator[TFTensor]:
    """
    Transfer VLImage to Tensorflow.

    Warning:
        use tensor in contextmanager lifecycle scope
    """
    if tf_to_dlpack is None:
        raise RuntimeError("tensorrt is not installed, minimal required version see in FSDK")
    coreImage = image.coreImage
    tfTensor = tf_from_dlpack(coreImage.__dlpack__())
    yield tfTensor


@contextlib.contextmanager
def fromTensorFlow(tfTensor: TFTensor) -> Iterator[VLImage]:
    """
    Transfer TensorFlow Tensor to VLImage without copy.

    Warning:
        use image in contextmanager lifecycle scope
    """
    coreImage = CoreImage.from_dlpack(tf_to_dlpack(tfTensor))
    yield VLImage(coreImage)
