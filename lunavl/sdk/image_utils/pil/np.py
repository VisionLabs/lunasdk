"""
Module contains helper functions for a pillow image conversion into np array
"""

import numpy as np
import PIL.Image
from PIL.Image import Image, _fromarray_typemap as imageTypeMap
from packaging.version import Version


def getNPImageType(arr: np.ndarray) -> str:
    """
    Get numpy image type
    Args:
        arr: numpy array
    Returns:
        image type which pillow associated with this array
    Raises:
        TypeError: if cannot handle  image type
    References:
        https://github.com/python-pillow/Pillow/blob/master/src/PIL/Image.py#L2788
    """
    try:
        typekey = (1, 1) + arr.shape[2:], arr.dtype.str
    except KeyError as e:
        raise TypeError("Cannot handle this data type: %s" % arr.dtype.str) from e
    try:
        imgType, *_ = imageTypeMap[typekey]
        return imgType
    except KeyError as e:
        raise TypeError("Cannot handle this data type: %s, %s" % typekey) from e


def pilToNumpy(img: Image) -> np.ndarray:
    """
    Fast load pillow image to numpy array
    Args:
        img: pillow image
    Returns:
        numpy array
    Raises:
        RuntimeError: if encoding failed
    References:
        https://habr.com/ru/post/545850/
        https://github.com/python-pillow/Pillow/pull/9504
    """
    img.load()
    # unpack data
    e = PIL.Image._getencoder(img.mode, "raw", img.mode)
    if Version(PIL.__version__) < Version("12.2.0"):
        e.setimage(img.im)
    else:
        e.setimage(img.im, None)

    # NumPy buffer for the result
    shape, typestr = PIL.Image._conv_type_shape(img)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)
        mem[offset : offset + len(d)] = d  # noqa: E203
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data
