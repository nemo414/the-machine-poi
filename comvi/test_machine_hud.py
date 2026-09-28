import csv
import unittest
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

import machine_hud


class MachineHudFallbackTests(unittest.TestCase):
    def test_classify_face_accepts_vector_variants(self):
        admin = np.linspace(0.0, 1.0, 128, dtype=float)
        same = admin.copy()
        label, distance = machine_hud.classify_face(admin, same, tolerance=0.5)
        self.assertEqual(label, "ADMIN")
        self.assertGreaterEqual(distance, 0.0)

    def test_classify_face_labels_non_match_unknown(self):
        admin = np.zeros(128, dtype=float)
        other = np.ones(128, dtype=float)
        label, distance = machine_hud.classify_face(admin, other, tolerance=0.5)
        self.assertEqual(label, "UNKNOWN")
        self.assertGreater(distance, 0.5)

    def test_classify_face_uses_closest_of_multiple_people_and_references(self):
        known = {
            "ALICE": [np.zeros(128), np.full(128, 0.2)],
            "BOB": [np.ones(128)],
        }
        label, distance = machine_hud.classify_face(known, np.full(128, 0.19), tolerance=0.5)
        self.assertEqual(label, "ALICE")
        self.assertLess(distance, 0.5)

        label, distance = machine_hud.classify_face(known, np.full(128, 2.0), tolerance=0.5)
        self.assertEqual(label, "UNKNOWN")
        self.assertGreater(distance, 0.5)

    def test_load_known_faces_reads_multiple_images_per_person(self):
        with TemporaryDirectory() as temporary_directory:
            faces_dir = Path(temporary_directory)
            (faces_dir / "ALICE").mkdir()
            (faces_dir / "ALICE" / "front.jpg").touch()
            (faces_dir / "ALICE" / "side.png").touch()
            (faces_dir / "BOB").mkdir()
            (faces_dir / "BOB" / "front.jpg").touch()
            encodings = [np.zeros(128), np.full(128, 0.2), np.ones(128)]
            with patch("machine_hud.load_admin_encoding", side_effect=encodings):
                known = machine_hud.load_known_faces(faces_dir)

        self.assertEqual(set(known), {"ALICE", "BOB"})
        self.assertEqual(len(known["ALICE"]), 2)

    def test_event_logger_writes_local_csv_and_deduplicates(self):
        with TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "events.csv"
            logger = machine_hud.EventLogger(log_path)
            detections = [
                ((0, 10, 10, 0), "ALICE", 0.1),
                ((20, 30, 30, 20), "ALICE", 0.2),
            ]
            logger.record(detections)
            logger.record(detections)
            with log_path.open(newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "ALICE")
        self.assertIn("T", rows[0]["timestamp"])

    def test_draw_screen_overlay_renders_detection_counts_and_log_status(self):
        frame = np.zeros((128, 320, 3), dtype=np.uint8)
        results = [((10, 20, 30, 0), "ALICE", 0.1), ((40, 50, 60, 30), "UNKNOWN", 1.2)]
        machine_hud.draw_screen_overlay(frame, "CAM-00-ABCD", 24.0, results, True)
        self.assertTrue(np.any(frame))

    def test_speech_worker_transcribes_audio_in_background(self):
        transcript_ready = threading.Event()
        fake_stream = Mock()
        fake_model = Mock()

        def transcribe(*_args, **_kwargs):
            transcript_ready.set()
            return [SimpleNamespace(text=" halo mesin ")], SimpleNamespace(language="id")

        fake_model.transcribe.side_effect = transcribe
        with patch("machine_hud.sd.RawInputStream", return_value=fake_stream) as stream_factory, \
                patch("machine_hud.WhisperModel", return_value=fake_model):
            worker = machine_hud.SpeechWorker(sample_rate=4, chunk_seconds=1)
            try:
                callback = stream_factory.call_args.kwargs["callback"]
                callback(np.zeros(4, dtype=np.int16).tobytes(), 4, None, None)
                self.assertTrue(transcript_ready.wait(2))
                self.assertEqual(worker.snapshot(), ("MENDENGARKAN", "halo mesin"))
                fake_model.transcribe.assert_called_once()
            finally:
                worker.close()

    def test_fallback_feature_vector_shape(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[20:100, 25:100] = 200
        vec = machine_hud.face_feature_vector(frame)
        self.assertEqual(vec.shape, (16384,))
        self.assertTrue(np.isfinite(vec).all())

    def test_fallback_detector_does_not_claim_identity(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cascade = unittest.mock.MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([[20, 20, 40, 40]])
        with patch("machine_hud.face_recognition", None), patch(
            "machine_hud.cv2.CascadeClassifier", return_value=cascade
        ):
            results = machine_hud.recognize_frame(
                frame, {"ADMIN": [np.zeros(16384)]}, 1.0, 0.5, 1
            )
        self.assertEqual(results[0][1], "UNKNOWN")

    def test_camera_read_retries_temporary_failures(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        capture = Mock()
        capture.read.side_effect = [(False, None), (False, None), (True, frame)]
        with patch("machine_hud.time.sleep") as sleep:
            result = machine_hud.read_camera_frame(capture, retries=3, retry_delay=0)
        self.assertIs(result, frame)
        self.assertEqual(capture.read.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_fallback_returns_no_faces_for_blank_frame(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        admin = np.zeros(16384, dtype=float)
        results = machine_hud.recognize_frame(frame, {"ADMIN": [admin]}, 0.25, 0.5, 1)
        self.assertEqual(results, [])

    def test_hud_draws_without_center_crosshair(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        machine_hud.draw_machine_hud(frame, 20, 100, 100, 20, (0, 0, 255), "UNKNOWN")
        self.assertFalse(np.any(frame[60, 60]))

    def test_inference_interval_defaults_to_four_and_is_configurable(self):
        self.assertEqual(machine_hud.parse_args([]).process_every, 4)
        self.assertEqual(machine_hud.parse_args(["--process-every", "2"]).process_every, 2)
        self.assertEqual(machine_hud.parse_args([]).inference_interval, 1.0)
        self.assertEqual(machine_hud.parse_args(["--inference-interval", "0.5"]).inference_interval, 0.5)
        self.assertEqual(machine_hud.parse_args([]).event_log, Path("events.csv"))
        self.assertTrue(machine_hud.parse_args(["--no-event-log"]).no_event_log)
        self.assertEqual(machine_hud.parse_args(["--known-dir", "faces"]).known_dir, Path("faces"))
        self.assertTrue(machine_hud.parse_args(["--listen"]).listen)

    def test_recognition_worker_publishes_background_result(self):
        completed = threading.Event()
        expected = [((10, 20, 30, 0), "ADMIN", 0.1)]

        def fake_recognition(*_args):
            completed.set()
            return expected

        with patch("machine_hud.recognize_frame", side_effect=fake_recognition):
            worker = machine_hud.RecognitionWorker(np.zeros(128), 0.25, 0.5, 1, False)
            try:
                worker.submit(np.zeros((32, 32, 3), dtype=np.uint8))
                self.assertTrue(completed.wait(2))
                self.assertEqual(worker.snapshot(), expected)
            finally:
                worker.close()


if __name__ == "__main__":
    unittest.main()
