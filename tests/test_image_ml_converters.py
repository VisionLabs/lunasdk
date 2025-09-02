import unittest

import numpy as np
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValue
from tensorflow import Tensor as TFTensor, config
from torch import Tensor as PytorchTensor

from lunavl.sdk.image_utils.image import MemoryResidence, VLImage
from lunavl.sdk.image_utils.transfers import toOnnx, toPytorch, toTensorFlow
from tests.base import BaseTestClass
from tests.resources import ONE_FACE


class TestImage(BaseTestClass):
    """
    Test of image conversation from/to ml framework tensors.
    """

    def assertNP(self, image, npArray: np.ndarray):
        assert image.rect.width == npArray.shape[1]
        assert image.rect.height == npArray.shape[0]
        assert image.getByteDepth == npArray.shape[2]
        assert np.array_equal(image.asNPArray(), npArray)

    def test_image_to_pytorch_cpu(self):
        image = VLImage.load(filename=ONE_FACE)
        with toPytorch(image) as tensor:
            assert isinstance(tensor, PytorchTensor)
            self.assertNP(image, tensor.numpy())

    def test_image_to_onnx_cpu(self):
        image = VLImage.load(filename=ONE_FACE)
        with toOnnx(image) as tensor:
            assert isinstance(tensor, OrtValue)
            self.assertNP(image, tensor.numpy())

    def test_image_to_onnx_gpu(self):
        image = VLImage.load(filename=ONE_FACE, memoryResidence=MemoryResidence.GPU)
        with toOnnx(image) as tensor:
            assert isinstance(tensor, OrtValue)
            assert tensor.device_name() == "Cuda"
            assert image.rect.width == tensor.shape()[1]
            assert image.rect.height == tensor.shape()[0]
            assert image.getByteDepth == tensor.shape()[2]

    def test_image_to_tf_cpu(self):
        image = VLImage.load(filename=ONE_FACE)
        with toTensorFlow(image) as tensor:
            assert isinstance(tensor, TFTensor)
            self.assertNP(image, tensor.numpy())

    @unittest.skip("need tensorflow[and-cuda]")
    def test_image_to_tf_gpu(self):
        image = VLImage.load(filename=ONE_FACE, memoryResidence=MemoryResidence.GPU)
        with toTensorFlow(image) as tensor:
            assert isinstance(tensor, TFTensor)

    @unittest.skip("need pytorch with cuda support")
    def test_image_to_pytorch_gpu(self):
        image = VLImage.load(filename=ONE_FACE)
        with toPytorch(image) as tensor:
            assert isinstance(tensor, PytorchTensor)
