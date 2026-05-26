"""
Unit tests for TrackEngineSettingsProvider.

Pattern per test:
    1. Create a fresh provider (loads trackengine.conf from $FSDK_ROOT/data).
    2. Write a new value and read it back — assert equality.

Enum properties and IntO1 properties are covered with parametrize so every
allowed value gets its own round-trip. All other properties are parametrized
with two representative values.
"""

import pytest

from lunavl.sdk.trackengine.setting_provider import (
    LoggingMode,
    OverlapRemovingType,
    TrackEngineSettingsProvider,
    TrackerType,
)


def makeProvider() -> TrackEngineSettingsProvider:
    return TrackEngineSettingsProvider()


# ── logging ──────────────────────────────────────────────────────────────────


class TestLoggingSettings:
    @pytest.mark.parametrize("mode", list(LoggingMode))
    def test_mode_roundtrip(self, mode):
        p = makeProvider()
        p.logging.mode = mode
        assert p.logging.mode == mode

    @pytest.mark.parametrize("value", ["custom.log", "other.log"])
    def test_log_file_path_roundtrip(self, value):
        p = makeProvider()
        p.logging.logFilePath = value
        assert p.logging.logFilePath == value

    @pytest.mark.parametrize("value", [0, 3])
    def test_severity_roundtrip(self, value):
        p = makeProvider()
        p.logging.severity = value
        assert p.logging.severity == value


# ── other ────────────────────────────────────────────────────────────────────


class TestOtherSettings:
    @pytest.mark.parametrize("value", [0, 1])
    def test_callback_mode_roundtrip(self, value):
        p = makeProvider()
        p.other.callbackMode = value
        assert p.other.callbackMode == value

    @pytest.mark.parametrize("value", [3, 15])
    def test_detector_step_roundtrip(self, value):
        p = makeProvider()
        p.other.detectorStep = value
        assert p.other.detectorStep == value

    @pytest.mark.parametrize("value", [0, 2])
    def test_detector_comparer_roundtrip(self, value):
        p = makeProvider()
        p.other.detectorComparer = value
        assert p.other.detectorComparer == value

    @pytest.mark.parametrize("value", [0, 1])
    def test_use_one_detection_mode_roundtrip(self, value):
        p = makeProvider()
        p.other.useOneDetectionMode = value
        assert p.other.useOneDetectionMode == value

    @pytest.mark.parametrize("value", [5, 30])
    def test_skip_frames_roundtrip(self, value):
        p = makeProvider()
        p.other.skipFrames = value
        assert p.other.skipFrames == value

    @pytest.mark.parametrize("value", [0, 1])
    def test_frg_subtractor_roundtrip(self, value):
        p = makeProvider()
        p.other.frgSubtractor = value
        assert p.other.frgSubtractor == value

    @pytest.mark.parametrize("value", [10, 50])
    def test_callback_buffer_size_roundtrip(self, value):
        p = makeProvider()
        p.other.callbackBufferSize = value
        assert p.other.callbackBufferSize == value

    @pytest.mark.parametrize("value", [10, 30])
    def test_tracking_results_buffer_size_roundtrip(self, value):
        p = makeProvider()
        p.other.trackingResultsBufferSize = value
        assert p.other.trackingResultsBufferSize == value

    @pytest.mark.parametrize("value", [0, 1])
    def test_detector_scaling_roundtrip(self, value):
        p = makeProvider()
        p.other.detectorScaling = value
        assert p.other.detectorScaling == value

    @pytest.mark.parametrize("value", [480, 1280])
    def test_scale_result_size_roundtrip(self, value):
        p = makeProvider()
        p.other.scaleResultSize = value
        assert p.other.scaleResultSize == value

    @pytest.mark.parametrize("value", [64, 256])
    def test_max_detection_count_roundtrip(self, value):
        p = makeProvider()
        p.other.maxDetectionCount = value
        assert p.other.maxDetectionCount == value

    @pytest.mark.parametrize("value", [2, 5])
    def test_minimal_track_length_roundtrip(self, value):
        p = makeProvider()
        p.other.minimalTrackLength = value
        assert p.other.minimalTrackLength == value

    @pytest.mark.parametrize("tracker_type", list(TrackerType))
    def test_tracker_type_roundtrip(self, tracker_type):
        p = makeProvider()
        p.other.trackerType = tracker_type
        assert p.other.trackerType == tracker_type

    @pytest.mark.parametrize("value", [0, 1])
    def test_kill_intersected_detections_roundtrip(self, value):
        p = makeProvider()
        p.other.killIntersectedDetections = value
        assert p.other.killIntersectedDetections == value

    @pytest.mark.parametrize("value", [0.3, 0.7])
    def test_kill_intersection_value_roundtrip(self, value):
        p = makeProvider()
        p.other.killIntersectionValue = value
        assert p.other.killIntersectionValue == pytest.approx(value, abs=1e-5)


# ── FRG ──────────────────────────────────────────────────────────────────────


class TestFRGSettings:
    @pytest.mark.parametrize("value", [0, 1])
    def test_use_binary_frg_roundtrip(self, value):
        p = makeProvider()
        p.frg.useBinaryFrg = value
        assert p.frg.useBinaryFrg == value

    @pytest.mark.parametrize("value", [10, 30])
    def test_frg_update_step_roundtrip(self, value):
        p = makeProvider()
        p.frg.frgUpdateStep = value
        assert p.frg.frgUpdateStep == value

    @pytest.mark.parametrize("value", [120, 200])
    def test_frg_scale_size_roundtrip(self, value):
        p = makeProvider()
        p.frg.frgScaleSize = value
        assert p.frg.frgScaleSize == value


# ── face tracking ─────────────────────────────────────────────────────────────


class TestFaceTrackingSettings:
    @pytest.mark.parametrize("value", [0, 1])
    def test_face_landmarks_detection_roundtrip(self, value):
        p = makeProvider()
        p.faceTracking.faceLandmarksDetection = value
        assert p.faceTracking.faceLandmarksDetection == value


# ── body tracking ─────────────────────────────────────────────────────────────


class TestBodyTrackingSettings:
    @pytest.mark.parametrize("value", [0, 1])
    def test_human_landmarks_detection_roundtrip(self, value):
        p = makeProvider()
        p.bodyTracking.humanLandmarksDetection = value
        assert p.bodyTracking.humanLandmarksDetection == value

    @pytest.mark.parametrize("strategy", list(OverlapRemovingType))
    def test_remove_overlapped_strategy_roundtrip(self, strategy):
        p = makeProvider()
        p.bodyTracking.removeOverlappedStrategy = strategy
        assert p.bodyTracking.removeOverlappedStrategy == strategy

    @pytest.mark.parametrize("value", [1.2, 2.0])
    def test_remove_horizontal_ratio_roundtrip(self, value):
        p = makeProvider()
        p.bodyTracking.removeHorizontalRatio = value
        assert p.bodyTracking.removeHorizontalRatio == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.3, 0.7])
    def test_iou_connection_threshold_roundtrip(self, value):
        p = makeProvider()
        p.bodyTracking.iouConnectionThreshold = value
        assert p.bodyTracking.iouConnectionThreshold == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0, 1])
    def test_use_body_reid_roundtrip(self, value):
        p = makeProvider()
        p.bodyTracking.useBodyReid = value
        assert p.bodyTracking.useBodyReid == value

    @pytest.mark.parametrize("value", [100, 120])
    def test_body_reid_version_roundtrip(self, value):
        p = makeProvider()
        p.bodyTracking.bodyReidVersion = value
        assert p.bodyTracking.bodyReidVersion == value

    @pytest.mark.parametrize("value", [0.7, 0.95])
    def test_reid_matching_threshold_roundtrip(self, value):
        p = makeProvider()
        p.bodyTracking.reidMatchingThreshold = value
        assert p.bodyTracking.reidMatchingThreshold == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [1, 5])
    def test_reid_matching_detections_count_roundtrip(self, value):
        p = makeProvider()
        p.bodyTracking.reidMatchingDetectionsCount = value
        assert p.bodyTracking.reidMatchingDetectionsCount == value


# ── detectors ─────────────────────────────────────────────────────────────────


class TestDetectorsSettings:
    @pytest.mark.parametrize("value", [0, 1])
    def test_use_face_detector_roundtrip(self, value):
        p = makeProvider()
        p.detectors.useFaceDetector = value
        assert p.detectors.useFaceDetector == value

    @pytest.mark.parametrize("value", [0, 1])
    def test_use_body_detector_roundtrip(self, value):
        p = makeProvider()
        p.detectors.useBodyDetector = value
        assert p.detectors.useBodyDetector == value


# ── HS-tracking ───────────────────────────────────────────────────────────────


class TestHSTrackingSettings:
    @pytest.mark.parametrize("value", [0.5, 0.9])
    def test_first_step_weight_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.firstStepWeight = value
        assert p.hsTracking.firstStepWeight == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.5, 0.9])
    def test_first_step_weight_human_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.firstStepWeightHuman = value
        assert p.hsTracking.firstStepWeightHuman == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.5, 0.9])
    def test_byte_step_weight_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.byteStepWeight = value
        assert p.hsTracking.byteStepWeight == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.5, 0.9])
    def test_byte_step_weight_human_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.byteStepWeightHuman = value
        assert p.hsTracking.byteStepWeightHuman == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.1, 0.4])
    def test_inertia_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.inertia = value
        assert p.hsTracking.inertia == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.02, 0.1])
    def test_inertia_human_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.inertiaHuman = value
        assert p.hsTracking.inertiaHuman == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.3, 0.6])
    def test_det_score_thr_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.detScoreThr = value
        assert p.hsTracking.detScoreThr == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.3, 0.6])
    def test_det_score_thr_human_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.detScoreThrHuman = value
        assert p.hsTracking.detScoreThrHuman == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.1, 0.4])
    def test_det_score_low_thr_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.detScoreLowThr = value
        assert p.hsTracking.detScoreLowThr == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.1, 0.4])
    def test_det_score_low_thr_human_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.detScoreLowThrHuman = value
        assert p.hsTracking.detScoreLowThrHuman == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.3, 0.6])
    def test_redet_score_thr_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.redetScoreThr = value
        assert p.hsTracking.redetScoreThr == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.3, 0.6])
    def test_redet_score_thr_human_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.redetScoreThrHuman = value
        assert p.hsTracking.redetScoreThrHuman == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.02, 0.1])
    def test_iou_thr_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.iouThr = value
        assert p.hsTracking.iouThr == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.1, 0.3])
    def test_iou_thr_human_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.iouThrHuman = value
        assert p.hsTracking.iouThrHuman == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.4, 0.8])
    def test_sot_tracker_score_thr_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.sotTrackerScoreThr = value
        assert p.hsTracking.sotTrackerScoreThr == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0, 1])
    def test_redetect_on_kalman_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.redetectOnKalman = value
        assert p.hsTracking.redetectOnKalman == value

    @pytest.mark.parametrize("value", [0, 1])
    def test_trust_low_assoc_roundtrip(self, value):
        p = makeProvider()
        p.hsTracking.trustLowAssoc = value
        assert p.hsTracking.trustLowAssoc == value


# ── experimental ──────────────────────────────────────────────────────────────


class TestExperimentalSettings:
    @pytest.mark.parametrize("value", [4, 8])
    def test_detect_max_batch_size_roundtrip(self, value):
        p = makeProvider()
        p.experimental.detectMaxBatchSize = value
        assert p.experimental.detectMaxBatchSize == value

    @pytest.mark.parametrize("value", [4, 8])
    def test_redetect_max_batch_size_roundtrip(self, value):
        p = makeProvider()
        p.experimental.redetectMaxBatchSize = value
        assert p.experimental.redetectMaxBatchSize == value

    @pytest.mark.parametrize("value", [4, 8])
    def test_tracker_max_batch_size_roundtrip(self, value):
        p = makeProvider()
        p.experimental.trackerMaxBatchSize = value
        assert p.experimental.trackerMaxBatchSize == value

    @pytest.mark.parametrize("value", [4, 8])
    def test_reid_max_batch_size_roundtrip(self, value):
        p = makeProvider()
        p.experimental.reidMaxBatchSize = value
        assert p.experimental.reidMaxBatchSize == value
