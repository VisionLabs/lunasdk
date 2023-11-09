from typing import Optional, Literal

from lunavl.sdk.faceengine.setting_provider import BaseSettingsProvider, BaseSettingsSection, BiDirectionEnum

IntO1 = Literal[0, 1]


class TrackerType(BiDirectionEnum):
    KCF = "kcf"
    OPENCV = "opencv"
    CARKALMAN = "carkalman"
    VL_TRACKER = "vlTracker"
    NONE = "none"


class OverlapRemovingType(BiDirectionEnum):
    BOTH = "both"
    SCORE = "score"
    NONE = "none"


class OtherSettings(BaseSettingsSection):
    """
    Descriptor factory settings.

    Properties:

        - model (int): CNN face descriptor version.
        - UseMobileNet (bool): mobile Net is faster but less accurate
        - distance (Distance): distance between descriptors on matching. L1 faster,L2 make better precision.
        - descriptorCountWarningLevel (float): Threshold,that limits the ratio of created  descriptors to the amount,
            defined by your liscence. Warning Level When the threshold is exceeded, FSDK prints the warning.

    """

    sectionName = "other"

    @property
    def callbackMode(self) -> Optional[IntO1]:
        """
        Getter for model

        Returns:
            model
        """
        return self.getValue("callback-mode")

    @callbackMode.setter
    def callbackMode(self, value: IntO1) -> None:
        """
        Setter for model
        Args:
            value: new value
        """
        self.setValue("callback-mode", value)

    @property
    def detectorStep(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("detector-step")

    @detectorStep.setter
    def detectorStep(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("detector-step", value)

    @property
    def detectorComparer(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("detector-comparer")

    @detectorComparer.setter
    def detectorComparer(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("detector-comparer", value)

    @property
    def useOneDetectionMode(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("use-one-detection-mode")

    @useOneDetectionMode.setter
    def useOneDetectionMode(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("use-one-detection-mode", value)

    @property
    def skipFrames(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("skip-frames")

    @skipFrames.setter
    def skipFrames(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("skip-frames", value)

    @property
    def frgSubtractor(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("frg-subtractor")

    @frgSubtractor.setter
    def frgSubtractor(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("frg-subtractor", value)

    @property
    def callbackBufferSize(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("callback-buffer-size")

    @callbackBufferSize.setter
    def callbackBufferSize(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("callback-buffer-size", value)

    @property
    def trackingResultsBufferSize(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("tracking-results-buffer-size")

    @trackingResultsBufferSize.setter
    def trackingResultsBufferSize(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("tracking-results-buffer-size", value)

    @property
    def detectorScaling(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("detector-scaling")

    @detectorScaling.setter
    def detectorScaling(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("detector-scaling", value)

    @property
    def scaleResultSize(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("scale-result-size")

    @scaleResultSize.setter
    def scaleResultSize(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("scale-result-size", value)

    @property
    def maxDetectionCount(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("max-detection-count")

    @maxDetectionCount.setter
    def maxDetectionCount(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("max-detection-count", value)

    @property
    def minimalTrackLength(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("minimal-track-length")

    @minimalTrackLength.setter
    def minimalTrackLength(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("minimal-track-length", value)

    @property
    def trackerType(self) -> Optional[TrackerType]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        trType = self.getValue("tracker-type")
        if trType:
            return TrackerType(trType)
        return None

    @trackerType.setter
    def trackerType(self, value: TrackerType) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("tracker-type", value.value)

    @property
    def killIntersectedDetections(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("kill-intersected-detections")

    @killIntersectedDetections.setter
    def killIntersectedDetections(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("kill-intersected-detections", value)

    @property
    def killIntersectionValue(self) -> Optional[float]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("kill-intersection-value")

    @killIntersectionValue.setter
    def killIntersectionValue(self, value: float) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("kill-intersection-value", value)


class FRGSettings(BaseSettingsSection):
    sectionName = "FRG"

    @property
    def useBinaryFrg(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("use-binary-frg")

    @useBinaryFrg.setter
    def useBinaryFrg(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("use-binary-frg", value)

    @property
    def frgUpdateStep(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("frg-update-step")

    @frgUpdateStep.setter
    def frgUpdateStep(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("frg-update-step", value)

    @property
    def frgScaleSize(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("frg-scale-size")

    @frgScaleSize.setter
    def frgScaleSize(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("frg-scale-size", value)


class FaceTrackingSettings(BaseSettingsSection):
    sectionName = "face"

    @property
    def faceLandmarksDetection(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("face-landmarks-detection")

    @faceLandmarksDetection.setter
    def faceLandmarksDetection(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("face-landmarks-detection", value)


class BodyTrackingSettings(BaseSettingsSection):
    sectionName = "human"

    @property
    def humanLandmarksDetection(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("human-landmarks-detection")

    @humanLandmarksDetection.setter
    def humanLandmarksDetection(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("human-landmarks-detection", value)

    @property
    def removeOverlappedStrategy(self) -> Optional[OverlapRemovingType]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        trType = self.getValue("remove-overlapped-strategy")
        if trType:
            return OverlapRemovingType(trType)
        return None

    @removeOverlappedStrategy.setter
    def removeOverlappedStrategy(self, value: OverlapRemovingType) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("remove-overlapped-strategy", value.value)

    @property
    def removeHorizontalRatio(self) -> Optional[float]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("remove-horizontal-ratio")

    @removeHorizontalRatio.setter
    def removeHorizontalRatio(self, value: float) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("remove-horizontal-ratio", value)

    @property
    def iouConnectionThreshold(self) -> Optional[float]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("iou-connection-threshold")

    @iouConnectionThreshold.setter
    def iouConnectionThreshold(self, value: float) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("iou-connection-threshold", value)

    @property
    def useBodyReid(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("use-body-reid")

    @useBodyReid.setter
    def useBodyReid(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("use-body-reid", value)

    @property
    def bodyReidVersion(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("body-reid-version")

    @bodyReidVersion.setter
    def bodyReidVersion(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("body-reid-version", value)

    @property
    def reidMatchingThreshold(self) -> Optional[float]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("reid-matching-threshold")

    @reidMatchingThreshold.setter
    def reidMatchingThreshold(self, value: float) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("reid-matching-threshold", value)

    @property
    def reidMatchingDetectionsCount(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("reid-matching-detections-count")

    @reidMatchingDetectionsCount.setter
    def reidMatchingDetectionsCount(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("reid-matching-detections-count", value)


class DetectorsSettings(BaseSettingsSection):
    sectionName = "detectors"

    @property
    def useFaceDetector(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("use-face-detector")

    @useFaceDetector.setter
    def useFaceDetector(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("use-face-detector", value)

    @property
    def useBodyDetector(self) -> Optional[IntO1]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("use-body-detector")

    @useBodyDetector.setter
    def useBodyDetector(self, value: IntO1) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("use-body-detector", value)


class ExperimentalSettings(BaseSettingsSection):
    sectionName = "experimental"

    @property
    def detectMaxBatchSize(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("detect-max-batch-size")

    @detectMaxBatchSize.setter
    def detectMaxBatchSize(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("detect-max-batch-size", value)

    @property
    def redetectMaxBatchSize(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("redetect-max-batch-size")

    @redetectMaxBatchSize.setter
    def redetectMaxBatchSize(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("redetect-max-batch-size", value)

    @property
    def trackerMaxBatchSize(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("tracker-max-batch-size")

    @trackerMaxBatchSize.setter
    def trackerMaxBatchSize(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("tracker-max-batch-size", value)

    @property
    def reidMaxBatchSize(self) -> Optional[int]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        return self.getValue("reid-max-batch-size")

    @reidMaxBatchSize.setter
    def reidMaxBatchSize(self, value: int) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("reid-max-batch-size", value)


class TrackEngineSettingsProvider(BaseSettingsProvider):
    # default configuration filename.
    defaultConfName = "trackengine.conf"

    @property
    def other(self) -> OtherSettings:
        return OtherSettings(self._coreSettingProvider)

    @property
    def faceTracking(self) -> FaceTrackingSettings:
        return FaceTrackingSettings(self._coreSettingProvider)

    @property
    def bodyTracking(self) -> BodyTrackingSettings:
        return BodyTrackingSettings(self._coreSettingProvider)

    @property
    def frg(self) -> FRGSettings:
        return FRGSettings(self._coreSettingProvider)

    @property
    def detectors(self) -> DetectorsSettings:
        return DetectorsSettings(self._coreSettingProvider)

    @property
    def experimenral(self) -> ExperimentalSettings:
        return ExperimentalSettings(self._coreSettingProvider)
