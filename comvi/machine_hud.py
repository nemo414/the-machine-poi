#!/usr/bin/env python3
"""The Machine HUD - Real-Time Face Classifier (Python 3.10+).

Instalasi (gunakan virtual environment):
    python -m pip install "setuptools<81"
    python -m pip install -r requirements.txt

Jalankan pada laptop dengan webcam dan desktop GUI:
    python machine_hud.py --admin admin_face.jpg
    python machine_hud.py --admin admin_face.jpg --camera 1 --mirror
    python machine_hud.py --admin admin_face.jpg --scale 0.5 --tolerance 0.5
    python machine_hud.py --known-dir known_faces --mirror

Q / ESC / tutup jendela: keluar. --help: seluruh opsi.
Gunakan opencv-python, BUKAN paket headless, untuk aplikasi webcam ini.
Jika dlib perlu dikompilasi, diperlukan CMake dan compiler C++ yang sesuai.
Windows tidak didukung resmi oleh face_recognition; instalasi bisa memerlukan
Visual Studio Build Tools dengan workload Desktop development with C++.

Foto referensi harus berisi tepat SATU wajah. Encoding 128 dimensi diekstrak
oleh model dlib yang sudah dilatih; program tidak melatih model baru.
UNKNOWN berarti wajah tidak cocok dengan referensi, bukan penilaian bahaya.
Ini demo pencocokan wajah, bukan autentikasi aman: tidak ada liveness / anti-spoof.
Frame dan encoding diproses lokal; log hanya menyimpan waktu dan label, bukan gambar.

Referensi API:
https://face-recognition.readthedocs.io/en/latest/face_recognition.html
https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
https://github.com/ageitgey/face_recognition#installation
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import queue
import secrets
import sys
import threading
import time

import cv2
import numpy as np

try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
except (ImportError, OSError):
    sd = None
    WhisperModel = None

try:
    import face_recognition  # type: ignore
except BaseException:  # pragma: no cover - optional dependency may be absent or incomplete
    face_recognition = None


# OpenCV memakai BGR, bukan RGB.
KNOWN_COLOR = (0, 255, 255)  # kuning
UNKNOWN_COLOR = (0, 0, 255)  # merah
UI_COLOR = (210, 225, 225)
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "THE MACHINE | FACE CLASSIFIER"


def draw_dashed_line(frame, start, end, color, dash=12, gap=8, thickness=2):
    """Garis putus-putus untuk arah apa pun, tanpa rectangle solid."""
    if dash <= 0 or gap <= 0 or thickness <= 0:
        raise ValueError("dash, gap, dan thickness harus positif.")
    x0, y0 = start
    dx, dy = end[0] - x0, end[1] - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return

    # Parameterisasi garis berdasarkan jarak s (piksel):
    # P(s) = P0 + (s/L) * (P1 - P0), dengan L = sqrt(dx^2 + dy^2).
    # Segmen ke-k dimulai pada s = k*(dash+gap), berakhir pada
    # min(s+dash, L). Daerah sepanjang gap dibiarkan tidak tergambar.
    for s in range(0, math.ceil(length), dash + gap):
        stop = min(s + dash, length)
        p = (round(x0 + dx * s / length), round(y0 + dy * s / length))
        q = (round(x0 + dx * stop / length), round(y0 + dy * stop / length))
        cv2.line(frame, p, q, color, thickness, cv2.LINE_AA)


def put_hud_text(frame, text, x, y, color, scale=0.55):
    """Teks dengan bayangan gelap; (x,y) merupakan baseline teks."""
    h, w = frame.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, 1)
    if tw > w - 8 and tw > 0:
        scale *= max(1, w - 8) / tw
        (tw, th), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x = max(2, min(int(x), w - tw - 3))
    y = max(th + 2, min(int(y), h - baseline - 3))
    # Ketebalan sama menjaga advance/jarak huruf kedua lapisan konsisten.
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def draw_machine_hud(frame, top, right, bottom, left, color, label):
    """HUD: empat sisi dashed, delapan garis sudut, crosshair, dan label.

    Koordinat mengikuti urutan face_recognition: top, right, bottom, left.
    Fungsi menggambar langsung pada frame BGR resolusi penuh.
    """
    height, width = frame.shape[:2]
    left, right = max(0, int(left)), min(width - 1, int(right))
    top, bottom = max(0, int(top)), min(height - 1, int(bottom))
    bw, bh = right - left, bottom - top
    if bw < 4 or bh < 4:
        return

    # Panjang kaki sudut proporsional terhadap sisi terpendek. Batas bw/3
    # dan bh/3 menjaga sudut berlawanan tidak bertemu pada kotak kecil.
    corner = max(1, min(28, max(8, round(min(bw, bh) * 0.16)), bw // 3, bh // 3))
    dash = max(4, min(12, round(min(bw, bh) * 0.06)))
    gap = max(3, round(dash * 0.65))
    thin = 1 if min(bw, bh) < 60 else 2
    thick = 2 if min(bw, bh) < 60 else 4

    # Bagian dashed hanya mengisi daerah di antara kaki-kaki sudut.
    sides = (
        ((left + corner, top), (right - corner, top)),
        ((left + corner, bottom), (right - corner, bottom)),
        ((left, top + corner), (left, bottom - corner)),
        ((right, top + corner), (right, bottom - corner)),
    )
    for start, end in sides:
        draw_dashed_line(frame, start, end, color, dash, gap, thin)

    # Sudut L: C + (sx*c, 0) dan C + (0, sy*c).
    # sx/sy menentukan arah menuju bagian dalam kotak dari setiap sudut.
    for x, y, sx, sy in (
        (left, top, 1, 1), (right, top, -1, 1),
        (left, bottom, 1, -1), (right, bottom, -1, -1),
    ):
        cv2.line(frame, (x, y), (x + sx * corner, y), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + sy * corner), color, thick, cv2.LINE_AA)

    # Pilih sisi label yang masih muat, lalu fallback ke atas/dalam kotak.
    cy = (top + bottom) // 2
    (text_width, text_height), _ = cv2.getTextSize(label, FONT, 0.6, 1)
    if right + 12 + text_width < width - 4:
        tx, ty = right + 12, cy + text_height // 2
    elif left - text_width - 12 >= 4:
        tx, ty = left - text_width - 12, cy + text_height // 2
    else:
        tx = left + 5
        ty = top - 10 if top > text_height + 14 else top + text_height + 9
    put_hud_text(frame, label, tx, ty, color, 0.6)


def validate_encoding(encoding):
    vector = np.asarray(encoding, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("Face encoding harus berisi nilai finite dan tidak kosong.")
    return vector


def face_feature_vector(frame):
    """Fitur fallback berbasis OpenCV untuk sistem tanpa face_recognition/dlib."""
    if frame is None or frame.size == 0:
        raise ValueError("Frame input kosong.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    vector = gray.astype(np.float64)
    vector /= 255.0
    return vector.reshape(-1)


def classify_face(known_encodings, candidate_encoding, tolerance=0.5):
    """Find the nearest enrolled reference; distance is not a confidence score."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance harus finite dan positif.")
    candidate = validate_encoding(candidate_encoding)
    if isinstance(known_encodings, dict):
        references_by_name = known_encodings
    else:
        references_by_name = {"ADMIN": [known_encodings]}

    closest_name = None
    closest_distance = math.inf
    for person_name, references in references_by_name.items():
        if not isinstance(references, (list, tuple)):
            references = [references]
        for reference in references:
            reference_vector = validate_encoding(reference)
            if reference_vector.shape != candidate.shape:
                raise ValueError("Dimensi encoding referensi dan kandidat harus sama.")
            distance = float(np.linalg.norm(reference_vector - candidate))
            if distance < closest_distance:
                closest_name = person_name
                closest_distance = distance

    if closest_name is None:
        raise ValueError("Belum ada wajah yang didaftarkan.")
    label = closest_name if closest_distance <= tolerance else "UNKNOWN"
    return label, closest_distance


def load_admin_encoding(path):
    """Validasi foto agar wajah pertama tidak dipilih secara ambigu."""
    if not path.is_file():
        raise FileNotFoundError(f"Foto Admin tidak ditemukan: {path}")

    if face_recognition is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Foto Admin tidak bisa dibaca: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            raise RuntimeError("Cascade detector wajah OpenCV tidak tersedia.")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) != 1:
            raise ValueError(
                f"Foto Admin harus berisi tepat satu wajah; terdeteksi {len(faces)}. "
                "Gunakan foto wajah yang jelas dan menghadap kamera."
            )
        x, y, w, h = faces[0]
        roi = img[y:y + h, x:x + w]
        return validate_encoding(face_feature_vector(roi))

    # face_recognition / dlib path
    rgb = face_recognition.load_image_file(str(path), mode="RGB")
    boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")
    if len(boxes) != 1:
        raise ValueError(
            f"Foto Admin harus berisi tepat satu wajah; terdeteksi {len(boxes)}. "
            "Gunakan foto wajah yang jelas dan menghadap kamera."
        )
    encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=1, model="small")
    if len(encodings) != 1:
        raise RuntimeError("Wajah referensi terdeteksi, tetapi encoding gagal.")
    return validate_encoding(encodings[0])


def load_known_faces(directory):
    """Load one or more reference photos from each named subdirectory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Folder wajah tidak ditemukan: {directory}")

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    known_encodings = {}
    for person_directory in sorted(path for path in directory.iterdir() if path.is_dir()):
        person_name = person_directory.name.strip()
        if not person_name or person_name.casefold() == "unknown":
            raise ValueError("Nama folder orang harus terisi dan tidak boleh 'UNKNOWN'.")
        image_paths = sorted(
            path for path in person_directory.iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        )
        if image_paths:
            known_encodings[person_name] = [
                load_admin_encoding(path) for path in image_paths
            ]

    if not known_encodings:
        raise ValueError(
            "Folder pendaftaran kosong. Buat subfolder per orang dan isi foto .jpg/.png."
        )
    return known_encodings


def restore_box(box, small_shape, original_shape, mirror=False):
    """Kembalikan koordinat ke resolusi asli, lalu refleksikan jika diminta."""
    sh, sw = small_shape[:2]
    height, width = original_shape[:2]
    sy, sx = height / sh, width / sw
    top, right, bottom, left = box
    # fx=0.25 tidak selalu berarti tepat kali empat karena pembulatan resize.
    # Gunakan faktor aktual untuk tiap sumbu, floor di awal dan ceil di akhir
    # agar cakupan bounding box tidak menyusut karena pembulatan.
    top = max(0, min(height - 1, math.floor(top * sy)))
    bottom = max(0, min(height - 1, math.ceil(bottom * sy)))
    left = max(0, min(width - 1, math.floor(left * sx)))
    right = max(0, min(width - 1, math.ceil(right * sx)))
    if mirror:
        # Refleksi posisi piksel: x' = (width-1) - x.
        left, right = width - 1 - right, width - 1 - left
    return top, right, bottom, left


def recognize_frame(frame, known_encodings, scale, tolerance, upsample, mirror=False):
    """Deteksi dan klasifikasi ulang setiap frame; tidak memakai label lama."""
    if face_recognition is None:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            raise RuntimeError("Cascade detector wajah OpenCV tidak tersedia.")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 0:
            return []
        results = []
        for (x, y, w, h) in faces:
            full_box = restore_box((y, x + w, y + h, x), small.shape, frame.shape, mirror)
            results.append((full_box, "UNKNOWN", math.inf))
        return results

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    # OpenCV: BGR. dlib: RGB uint8 contiguous. cvtColor menghindari
    # negative-stride view yang dapat terjadi dengan frame[:, :, ::-1].
    rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=upsample,
                                            model="hog")
    if not boxes:
        return []
    encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=1, model="small")
    if len(encodings) != len(boxes):
        raise RuntimeError("Jumlah encoding tidak sesuai deteksi; klasifikasi dihentikan.")
    results = []
    for box, encoding in zip(boxes, encodings):
        label, distance = classify_face(known_encodings, encoding, tolerance)
        full_box = restore_box(box, small.shape, frame.shape, mirror)
        results.append((full_box, label, distance))
    return results


class RecognitionWorker:
    def __init__(self, known_encodings, scale, tolerance, upsample, mirror):
        self._known_encodings = known_encodings
        self._scale = scale
        self._tolerance = tolerance
        self._upsample = upsample
        self._mirror = mirror
        self._frames = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._results = []
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame):
        latest_frame = frame.copy()
        try:
            self._frames.put_nowait(latest_frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(latest_frame)
            except queue.Full:
                pass

    def snapshot(self):
        with self._lock:
            results, error = self._results.copy(), self._error
        if error is not None:
            raise RuntimeError(f"Inference wajah gagal: {error}") from error
        return results

    def close(self):
        while True:
            try:
                self._frames.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
        self._thread.join()

    def _run(self):
        while True:
            frame = self._frames.get()
            if frame is None:
                return
            try:
                results = recognize_frame(frame, self._known_encodings, self._scale,
                                          self._tolerance, self._upsample, self._mirror)
            except Exception as exc:
                with self._lock:
                    self._error = exc
                return
            with self._lock:
                self._results = results


class EventLogger:
    def __init__(self, path, cooldown_seconds=30.0):
        if not math.isfinite(cooldown_seconds) or cooldown_seconds <= 0:
            raise ValueError("Jeda log harus finite dan positif.")
        self._path = Path(path)
        self._cooldown_seconds = cooldown_seconds
        self._last_logged = {}

    def record(self, results):
        now_monotonic = time.monotonic()
        labels = sorted({label for _box, label, _distance in results})
        ready_labels = [
            label for label in labels
            if now_monotonic - self._last_logged.get(label, -math.inf) >= self._cooldown_seconds
        ]
        if not ready_labels:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            if needs_header:
                writer.writerow(("timestamp", "label"))
            for label in ready_labels:
                writer.writerow((timestamp, label))
                self._last_logged[label] = now_monotonic


class SpeechWorker:
    def __init__(self, model_name="tiny", device=None, sample_rate=16000, chunk_seconds=4):
        if sd is None or WhisperModel is None:
            raise RuntimeError("Fitur suara perlu faster-whisper dan sounddevice di .venv.")
        if sample_rate <= 0 or chunk_seconds <= 0:
            raise ValueError("Sample rate dan panjang potongan audio harus positif.")
        try:
            self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as exc:
            raise RuntimeError(f"Model suara tidak dapat dimuat/diunduh: {exc}") from exc
        self._audio_queue = queue.Queue(maxsize=16)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._status = "MENYIAPKAN"
        self._last_text = ""
        self._sample_rate = sample_rate
        self._chunk_samples = sample_rate * chunk_seconds
        self._stream = sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=sample_rate,
            device=device,
            dtype="int16",
            channels=1,
            callback=self._capture_audio,
        )
        try:
            self._stream.start()
        except Exception as exc:
            self._stream.close()
            raise RuntimeError(f"Mikrofon tidak dapat dibuka: {exc}") from exc

        with self._lock:
            self._status = "MENDENGARKAN"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _capture_audio(self, audio_data, _frames, _time_info, status):
        if status or self._stop_event.is_set():
            return
        try:
            self._audio_queue.put_nowait(bytes(audio_data))
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(bytes(audio_data))
            except queue.Full:
                pass

    def _run(self):
        audio_chunks = []
        samples_collected = 0
        while not self._stop_event.is_set():
            try:
                audio_data = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_chunks.append(samples)
                samples_collected += samples.size
                if samples_collected < self._chunk_samples:
                    continue

                audio = np.concatenate(audio_chunks)
                audio_chunks.clear()
                samples_collected = 0
                with self._lock:
                    self._status = "MEMPROSES"
                segments, _info = self._model.transcribe(
                    audio,
                    language="id",
                    beam_size=1,
                    condition_on_previous_text=False,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                with self._lock:
                    if text:
                        self._last_text = text
                    self._status = "MENDENGARKAN"
            except Exception as exc:
                with self._lock:
                    self._status = "ERROR"
                    self._last_text = str(exc)[:80]
                return

    def snapshot(self):
        with self._lock:
            return self._status, self._last_text

    def close(self):
        self._stop_event.set()
        try:
            self._stream.stop()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass
        self._thread.join(timeout=2.0)


def draw_screen_overlay(frame, camera_id, fps, results, logging_enabled,
                        speech_status="OFF", speech_text=""):
    h, w = frame.shape[:2]
    known_count = sum(label != "UNKNOWN" for _box, label, _distance in results)
    unknown_count = len(results) - known_count
    log_status = "LOG ON" if logging_enabled else "LOG OFF"
    # ID dibuat sekali per sesi; hanya timestamp yang berubah tiap detik.
    put_hud_text(frame, "THE MACHINE // VISUAL ANALYSIS", 16, 28, UI_COLOR, 0.6)
    put_hud_text(frame, f"{camera_id}  |  {datetime.now():%H:%M:%S}", 16, 53, UI_COLOR)
    cv2.line(frame, (16, 64), (min(w - 16, 380), 64), UI_COLOR, 1, cv2.LINE_AA)
    if speech_status != "OFF":
        speech_display = speech_text or "MENUNGGU UCAPAN"
        put_hud_text(frame, f"MIC {speech_status}  |  {speech_display}", 16, 88,
                     UI_COLOR, 0.5)
    put_hud_text(frame, f"FPS {fps:05.1f}  |  KNOWN {known_count:02d}  |  UNKNOWN {unknown_count:02d}  |  {log_status}  |  Q / ESC",
                 16, h - 18, UI_COLOR, 0.5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--admin", type=Path, help="Foto referensi tunggal untuk mode Admin")
    parser.add_argument("--known-dir", type=Path,
                        help="Folder berisi satu subfolder foto untuk tiap orang")
    parser.add_argument("--camera", type=int, default=0, help="Indeks webcam, biasanya 0")
    parser.add_argument("--width", type=int, default=1280, help="Resolusi yang diminta")
    parser.add_argument("--height", type=int, default=720, help="Resolusi yang diminta")
    parser.add_argument("--scale", type=float, default=0.25, help="Skala inferensi (0,1]")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Ambang jarak embedding")
    parser.add_argument("--upsample", type=int, choices=(0, 1, 2), default=1,
                        help="0 lebih cepat; 1/2 membantu deteksi wajah kecil")
    parser.add_argument("--process-every", type=int, default=4,
                        help="Jalankan pengenalan tiap N frame; frame lain memakai hasil terakhir")
    parser.add_argument("--inference-interval", type=float, default=1.0,
                        help="Jeda minimum antar-inferensi dalam detik")
    parser.add_argument("--event-log", type=Path, default=Path("events.csv"),
                        help="CSV lokal; satu event per label tiap 30 detik")
    parser.add_argument("--no-event-log", action="store_true",
                        help="Nonaktifkan pencatatan kejadian lokal")
    parser.add_argument("--listen", action="store_true",
                        help="Aktifkan transkripsi suara Bahasa Indonesia secara lokal")
    parser.add_argument("--speech-model", choices=("tiny", "base"), default="tiny",
                        help="Model Whisper lokal (tiny lebih ringan; base lebih akurat)")
    parser.add_argument("--speech-device", type=int,
                        help="Indeks mikrofon sounddevice; default mengikuti perangkat Windows")
    parser.add_argument("--mirror", action="store_true", help="Cerminkan tampilan webcam")
    args = parser.parse_args(argv)
    if args.admin is not None and args.known_dir is not None:
        parser.error("Gunakan --admin atau --known-dir, jangan keduanya.")
    if not math.isfinite(args.scale) or not 0 < args.scale <= 1:
        parser.error("--scale harus finite dalam rentang (0, 1].")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance harus finite dan positif.")
    if args.process_every <= 0:
        parser.error("--process-every harus positif.")
    if not math.isfinite(args.inference_interval) or args.inference_interval <= 0:
        parser.error("--inference-interval harus finite dan positif.")
    if args.width <= 0 or args.height <= 0 or args.camera < 0:
        parser.error("--width/--height harus positif; --camera harus >= 0.")
    if args.speech_device is not None and args.speech_device < 0:
        parser.error("--speech-device harus >= 0.")
    return args


def read_camera_frame(capture, retries=10, retry_delay=0.1):
    for attempt in range(retries):
        ok, frame = capture.read()
        if ok and frame is not None and frame.size > 0:
            return frame
        if attempt + 1 < retries:
            time.sleep(retry_delay)
    return None


def run(args):
    if args.known_dir is not None:
        known_encodings = load_known_faces(args.known_dir.expanduser())
    else:
        admin_path = args.admin or Path("admin_face.jpg")
        known_encodings = {"ADMIN": [load_admin_encoding(admin_path.expanduser())]}
    event_logger = None if args.no_event_log else EventLogger(args.event_log.expanduser())
    camera_id = f"CAM-{args.camera:02d}-{secrets.token_hex(2).upper()}"
    cap = cv2.VideoCapture(args.camera)
    worker = None
    speech_worker = None
    try:
        if not cap.isOpened():
            raise RuntimeError("Webcam tidak dapat dibuka. Cek izin kamera atau --camera 1.")
        # Backend boleh mengabaikan properti ini; dimensi frame aktual tetap
        # menjadi dasar perhitungan, bukan angka resolusi yang diminta.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        worker = RecognitionWorker(known_encodings, args.scale, args.tolerance,
                       args.upsample, args.mirror)
        if args.listen:
            speech_worker = SpeechWorker(args.speech_model, args.speech_device)
        fps, previous = 0.0, time.perf_counter()
        next_inference = 0.0
        frame_number, results = 0, []
        print(f"{len(known_encodings)} identitas dimuat. Log lokal: "
              f"{args.event_log.expanduser() if event_logger else 'nonaktif'}.")
        if speech_worker is not None:
            print("Mikrofon aktif. Ucapan diproses lokal dan tidak disimpan.")
        print("Tekan Q atau ESC untuk keluar.")
        while True:
            frame = read_camera_frame(cap)
            if frame is None:
                raise RuntimeError("Frame webcam gagal dibaca; periksa koneksi/izin kamera.")

            # Encoding dihitung pada frame asli. Hanya preview yang dicerminkan
            # agar orientasi input model konsisten dengan foto referensi.
            now = time.perf_counter()
            if (frame_number % args.process_every == 0
                    and now >= next_inference):
                worker.submit(frame)
                next_inference = now + args.inference_interval
            frame_number += 1
            results = worker.snapshot()
            if event_logger is not None:
                event_logger.record(results)
            speech_status, speech_text = (
                speech_worker.snapshot() if speech_worker is not None else ("OFF", "")
            )
            display = cv2.flip(frame, 1) if args.mirror else frame.copy()
            for box, label, _distance in results:
                color = UNKNOWN_COLOR if label == "UNKNOWN" else KNOWN_COLOR
                draw_machine_hud(display, *box, color, label)

            # FPS dihitung antar-iterasi termasuk capture/inferensi/render
            # sebelumnya. EMA mengurangi lonjakan angka pada tampilan.
            now = time.perf_counter()
            instantaneous = 1.0 / max(now - previous, 1e-9)
            fps = instantaneous if fps == 0 else 0.9 * fps + 0.1 * instantaneous
            previous = now
            draw_screen_overlay(display, camera_id, fps, results, event_logger is not None,
                                speech_status, speech_text)
            cv2.imshow(WINDOW, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if speech_worker is not None:
            speech_worker.close()
        if worker is not None:
            worker.close()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Jangan menutupi error utama bila GUI tidak tersedia.


def main():
    args = parse_args()
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Pastikan dependensi terpasang dan jalankan di desktop lokal dengan GUI. "
              "Pakai opencv-python untuk tampilan webcam.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
