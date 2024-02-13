"""
Module contains a face descriptor estimator

See `face descriptor`_.

"""

from typing import List, Optional, Tuple, Union

from FaceEngine import IDescriptorExtractorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.descriptors.descriptors import FaceDescriptor, FaceDescriptorBatch, FaceDescriptorFactory

from ...launch_options import LaunchOptions
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import estimate, estimateDescriptorsBatch
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage

FaceDescriptorBatchEstimation = Tuple[FaceDescriptorBatch, Union[FaceDescriptor, None]]


class FaceDescriptorEstimator(BaseEstimator):
    """
    Face descriptor estimator.
    """

    #  pylint: disable=W0235
    def __init__(
        self,
        coreExtractor: IDescriptorExtractorPtr,
        faceDescriptorFactory: "FaceDescriptorFactory",
        launchOptions: LaunchOptions,
    ):
        """
        Init.

        Args:
            faceDescriptorFactory: face descriptor factory
        """
        super().__init__(coreExtractor, launchOptions)
        self.descriptorFactory = faceDescriptorFactory

    #  pylint: disable=W0221
    def estimate(  # type: ignore
        self,
        warp: Union[FaceWarp, FaceWarpedImage],
        descriptor: Optional[FaceDescriptor] = None,
        asyncEstimate: bool = False,
    ) -> Union[FaceDescriptor, AsyncTask[FaceDescriptor]]:
        """
        Estimate face descriptor from a warp image.

        Args:
            warp: warped image
            descriptor: descriptor for saving extract result
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated descriptor if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        return estimate(  # type: ignore
            warp=warp,
            descriptor=descriptor,
            descriptorFactory=self.descriptorFactory,
            coreEstimator=self._coreEstimator,
            asyncEstimate=asyncEstimate,
        )

    def estimateDescriptorsBatch(
        self,
        warps: List[Union[FaceWarp, FaceWarpedImage]],
        aggregate: bool = False,
        descriptorBatch: Optional[FaceDescriptorBatch] = None,
        asyncEstimate: bool = False,
    ) -> Union[FaceDescriptorBatchEstimation, AsyncTask[FaceDescriptorBatchEstimation]]:
        """
        Estimate a batch of descriptors from warped images.

        Args:
            warps: warped images
            aggregate:  whether to estimate  aggregate descriptor or not
            descriptorBatch: optional batch for saving descriptors
            asyncEstimate: estimate or run estimation in background
        Returns:
            tuple of batch and the aggregate descriptors (or None) if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed

        """
        return estimateDescriptorsBatch(  # type: ignore
            warps=warps,
            descriptorFactory=self.descriptorFactory,  # type: ignore
            aggregate=aggregate,
            descriptorBatch=descriptorBatch,
            coreEstimator=self._coreEstimator,
            asyncEstimate=asyncEstimate,
        )
