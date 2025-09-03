import random
import unittest

import numpy as np
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValue
from tensorflow import Tensor as TFTensor
from torch import Tensor as PytorchTensor

from lunavl.sdk.image_utils.image import MemoryResidence, VLImage
from lunavl.sdk.image_utils.transfers import fromOnnx, fromPytorch, fromTensorFlow, toOnnx, toPytorch, toTensorFlow
from tests.base import BaseTestClass
from tests.resources import ONE_FACE


class TestImage(BaseTestClass):
    """
    Test of image conversation from/to ml framework tensors.
    """

    def _assertNP(self, image: VLImage, npArray: np.ndarray):
        """
        Assert that an image and numpy array from tensor same.
        """
        assert image.rect.width == npArray.shape[1]
        assert image.rect.height == npArray.shape[0]
        assert image.getByteDepth == npArray.shape[2]
        assert np.array_equal(image.asNPArray(), npArray)
        # сhecking that the memory is really the same (reference to the memory match)
        value = random.randint(1, 256)
        image.asNPArray()[0][1][0] = value
        assert npArray[0][1][0] == value

    def test_image_to_pytorch_cpu(self):
        """
        Test transfer image to torch tensor on cpu
        """
        image = VLImage.load(filename=ONE_FACE)
        with toPytorch(image) as tensor:
            assert isinstance(tensor, PytorchTensor)
            self._assertNP(image, tensor.numpy())
            with fromPytorch(tensor) as vlImage:
                assert isinstance(vlImage, VLImage)
                self._assertNP(vlImage, tensor.numpy())

    @unittest.skip("need pytorch with cuda support")
    def test_image_to_pytorch_gpu(self):
        """
        Test transfer image to torch tensor on gpu
        """
        image = VLImage.load(filename=ONE_FACE)
        with toPytorch(image) as tensor:
            assert isinstance(tensor, PytorchTensor)
            with fromPytorch(tensor) as vlImage:
                assert isinstance(vlImage, VLImage)

    def test_image_to_onnx_cpu(self):
        """
        Test transfer image to onnx tensor on cpu
        """
        image = VLImage.load(filename=ONE_FACE)
        with toOnnx(image) as tensor:
            assert isinstance(tensor, OrtValue)
            self._assertNP(image, tensor.numpy())
            with fromOnnx(tensor) as vlImage:
                assert isinstance(vlImage, VLImage)
                self._assertNP(vlImage, tensor.numpy())

    def test_image_to_onnx_gpu(self):
        """
        Test transfer image to onnx tensor on gpu
        """
        image = VLImage.load(filename=ONE_FACE, memoryResidence=MemoryResidence.GPU)
        with toOnnx(image) as tensor:
            assert isinstance(tensor, OrtValue)
            assert tensor.device_name() == "Cuda"
            assert image.rect.width == tensor.shape()[1]
            assert image.rect.height == tensor.shape()[0]
            assert image.getByteDepth == tensor.shape()[2]
            with fromOnnx(tensor) as vlImage:
                assert isinstance(vlImage, VLImage)
                assert vlImage.rect.width == tensor.shape()[1]
                assert vlImage.rect.height == tensor.shape()[0]
                assert vlImage.getByteDepth == tensor.shape()[2]
                assert vlImage.getMemoryResidence() == MemoryResidence.GPU

    def test_image_to_tf_cpu(self):
        """
        Test transfer image to tensorflow tensor on cpu
        """
        image = VLImage.load(filename=ONE_FACE)
        with toTensorFlow(image) as tensor:
            assert isinstance(tensor, TFTensor)
            self._assertNP(image, tensor.numpy())
            with fromTensorFlow(tensor) as vlImage:
                assert isinstance(vlImage, VLImage)
                self._assertNP(vlImage, tensor.numpy())

    @unittest.skip("need tensorflow[and-cuda]")
    def test_image_to_tf_gpu(self):
        """
        Test transfer image to tensorflow tensor on gpu
        """
        image = VLImage.load(filename=ONE_FACE, memoryResidence=MemoryResidence.GPU)
        with toTensorFlow(image) as tensor:
            assert isinstance(tensor, TFTensor)
            with fromTensorFlow(tensor) as vlImage:
                assert isinstance(vlImage, VLImage)
