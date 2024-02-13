"""
Module contains a face descriptor estimator

See `face descriptor`_.

"""

from typing import Dict, Iterator, List, Optional, Type, Union

from FaceEngine import DescriptorBatchResult, IDescriptorBatchPtr, IDescriptorPtr  # pylint: disable=E0611,E0401

from ..base import BaseEstimation
from ..errors.exceptions import assertError
from ..globals import DEFAULT_HUMAN_DESCRIPTOR_VERSION as DHDV


class BaseDescriptor(BaseEstimation):
    """
    Descriptor

    Attributes:
        garbageScore (float): garbage score
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: IDescriptorPtr, garbageScore: float = 0.0):
        super().__init__(coreEstimation)
        self.garbageScore = garbageScore

    def asDict(self) -> Dict[str, Union[float, bytes]]:
        """
        Convert to dict

        Returns:
            Dict with keys "descriptor" and "score"
        """
        return {"descriptor": self.coreEstimation.getData(), "score": self.garbageScore, "version": self.model}

    @property
    def rawDescriptor(self) -> bytes:
        """
        Get raw descriptors
        Returns:
            bytes with metadata
        """
        error, descBytes = self.coreEstimation.save()
        assertError(error)
        return descBytes

    @property
    def asVector(self) -> List[int]:
        """
        Convert descriptor to list of ints
        Returns:
            list of ints.
        """
        return self.coreEstimation.getDescriptor()

    @property
    def asBytes(self) -> bytes:
        """
        Get descriptor as bytes.

        Returns:

        """
        return self.coreEstimation.getData()

    @property
    def model(self) -> int:
        """
        Get model of descriptor
        Returns:
            model version
        """
        return self.coreEstimation.getModelVersion()

    def reload(self, descriptor: bytes, garbageScore: float = 0.0) -> None:
        """
        Reload internal descriptor bytes.

        Args:
            descriptor: descriptor bytes
            garbageScore: new garbage scores

        Raises:
            LunaSDKException(LunaVLError.fromSDKError(res)) if cannot create descriptor instance
        """
        error = self.coreEstimation.load(descriptor, len(descriptor))
        assertError(error)
        self.garbageScore = garbageScore


class BaseDescriptorBatch(BaseEstimation):
    """
    Base descriptor batch.

    Attributes:
        scores (List[float]):  list of garbage scores
    """

    _descriptorFactory: Type[BaseDescriptor]

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: IDescriptorBatchPtr, scores: Optional[List[float]] = None):
        super().__init__(coreEstimation)
        if scores is None:
            self.scores = [0.0 for _ in range(coreEstimation.getMaxCount())]
        else:
            self.scores = scores

    def __len__(self) -> int:
        """
        Get descriptors count.

        Returns:
            descriptors count
        """
        return self._coreEstimation.getCount()

    def maxLen(self) -> int:
        """
        Get batch size.

        Returns:
            batch size
        """
        return self._coreEstimation.getMaxCount()

    def asDict(self) -> List[Dict]:
        """
        Get batch in json like object.

        Returns:
            list of descriptors dict
        """
        return [descriptor.asDict() for descriptor in self]

    def __getitem__(self, i) -> BaseDescriptor:
        """
        Get descriptor by index

        Args:
            i: index

        Returns:
            descriptor
        """
        if i >= len(self):
            raise IndexError(f"Descriptor index '{i}' out of range")  # todo remove after
        error, descriptor = self._coreEstimation.getDescriptorFast(i)
        assertError(error)
        descriptor = self.__class__._descriptorFactory(descriptor, self.scores[i])
        return descriptor

    def __iter__(self) -> Iterator[BaseDescriptor]:
        """
        Iterator by batch.

        Yields:
            descriptors
        """
        for index in range(len(self)):
            error, descriptor = self._coreEstimation.getDescriptorFast(index)
            assertError(error)
            yield self._descriptorFactory(descriptor, self.scores[index])

    def append(self, descriptor: BaseDescriptor) -> None:
        """
        Add descriptor to end of batch.

        Args:
            descriptor: descriptor
        """
        error: DescriptorBatchResult = self.coreEstimation.add(descriptor.coreEstimation)
        assertError(error)
        self.scores.append(descriptor.garbageScore)

    def __repr__(self) -> str:
        """
        Representation.

        Returns:
            str(self.asDict())
        """
        fullDescriptors = self.asDict()
        for d in fullDescriptors:
            del d["descriptor"]
        return str(fullDescriptors)


class BaseDescriptorFactory:
    """
    Base Descriptor factory.

    Attributes:
        _faceEngine (VLFaceEngine): faceEngine
        _descriptorVersion (int): descriptor version or zero for use default descriptor version
    """

    _descriptorFactory: Type[BaseDescriptor]
    _descriptorBatchFactory: Type[BaseDescriptorBatch]

    def __init__(self, faceEngine: "VLFaceEngine", descriptorVersion: int = 0):  # type: ignore # noqa: F821
        self._faceEngine = faceEngine
        self._descriptorVersion = descriptorVersion

    @property
    def descriptorVersion(self) -> int:
        """
        Return descriptor version for generating descriptor
        Returns:
            _descriptorVersion
        """
        return self._descriptorVersion

    def generateDescriptor(
        self, descriptor: Optional[bytes] = None, garbageScore: Optional[float] = None, descriptorVersion=0
    ) -> BaseDescriptor:
        """
        Generate core descriptor.

        Args:
            descriptor: the input descriptor
            garbageScore: the input descriptor garbage score
            descriptorVersion: descriptor version or zero for use default descriptor version

        Returns:
            a core descriptor

        Raises:
            ValueError if garbageScore is not empty and descriptor is empty
        """
        if garbageScore is not None and descriptor is None:
            raise ValueError("Do not specify `garbageScore` unexpected")

        if descriptor is not None:
            version = int.from_bytes(descriptor[4:8], byteorder="little")
            outputDescriptor = self.__class__._descriptorFactory(
                self._faceEngine.coreFaceEngine.createDescriptor(version)
            )
            if garbageScore is not None:
                outputDescriptor.reload(descriptor=descriptor, garbageScore=garbageScore)
            else:
                outputDescriptor.reload(descriptor=descriptor)
        else:
            outputDescriptor = self.__class__._descriptorFactory(
                self._faceEngine.coreFaceEngine.createDescriptor(descriptorVersion or self._descriptorVersion)
            )
        return outputDescriptor

    def generateDescriptorsBatch(self, size: int, descriptorVersion: int = 0) -> BaseDescriptorBatch:
        """
        Generate core descriptors batch.

        Args:
            size: maximum batch size
            descriptorVersion: descriptor version or zero for use default descriptor version

        Returns:
            empty batch with the restricted maximum size
        """
        descriptorVersion = descriptorVersion or self._descriptorVersion
        coreBatch: IDescriptorBatchPtr = self._faceEngine.coreFaceEngine.createDescriptorBatch(
            size, version=descriptorVersion
        )
        return self._descriptorBatchFactory(coreBatch)


class FaceDescriptor(BaseDescriptor):
    """
    Face Descriptor class
    """

    pass


class FaceDescriptorBatch(BaseDescriptorBatch):
    """
    Face descriptor batch.
    """

    _descriptorFactory = FaceDescriptor


class FaceDescriptorFactory(BaseDescriptorFactory):
    """
    Face Descriptor factory.
    """

    _descriptorBatchFactory = FaceDescriptorBatch
    _descriptorFactory = FaceDescriptor


class BodyDescriptor(BaseDescriptor):
    """
    Body Descriptor class
    """

    pass


class BodyDescriptorBatch(BaseDescriptorBatch):
    """
    Body descriptor batch.
    """

    _descriptorFactory = BodyDescriptor


class BodyDescriptorFactory(BaseDescriptorFactory):
    """
    Body Descriptor factory.
    """

    def __init__(self, faceEngine: "VLFaceEngine", descriptorVersion: int = DHDV):  # type: ignore # noqa: F821
        super().__init__(faceEngine, descriptorVersion)

    _descriptorBatchFactory = BodyDescriptorBatch
    _descriptorFactory = BodyDescriptor
