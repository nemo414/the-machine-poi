# The Machine HUD

Demo pengenalan wajah lokal dengan webcam. Wajah tidak cocok ditandai `UNKNOWN`, bukan dianggap ancaman.

## Satu foto referensi

Jalankan dari root proyek:

```powershell
& ".\.venv\Scripts\python.exe" .\comvi\machine_hud.py --admin .\comvi\admin_face.jpg --mirror
```

## Beberapa orang

Buat folder per orang di `comvi/known_faces`, lalu taruh satu atau beberapa foto `.jpg`, `.jpeg`, `.png`, atau `.bmp` di tiap folder. Setiap foto harus memuat tepat satu wajah.

```text
comvi/known_faces/
  ALICE/
    depan.jpg
    samping.jpg
  BOB/
    depan.png
```

Nama folder menjadi label HUD dan label di log. Gunakan nama samaran bila nama asli tidak diperlukan. Minta persetujuan orang yang fotonya dipakai; foto referensi tetap tersimpan di folder tersebut.

```powershell
& ".\.venv\Scripts\python.exe" .\comvi\machine_hud.py --known-dir .\comvi\known_faces --mirror
```

## Log dan akurasi

Secara default aplikasi membuat `events.csv` di direktori kerja saat ada wajah terdeteksi. Isinya hanya waktu lokal dan label, tanpa foto; label yang sama dicatat paling sering sekali tiap 30 detik. Ubah lokasi dengan `--event-log .\comvi\events.csv` atau nonaktifkan dengan `--no-event-log`.

Beberapa foto dengan variasi pencahayaan dan sudut dapat membantu pencocokan, tetapi tidak menjamin identifikasi akurat. `--tolerance` mengatur ambang jarak; nilai lebih kecil lebih ketat. Uji dengan foto terpisah, dan jangan memperlakukan jarak sebagai persentase keyakinan. Jika `face_recognition`/`dlib` tidak tersedia, sistem hanya mendeteksi wajah dan memberi label `UNKNOWN`.

## Transkripsi suara

Opsi `--listen` mengaktifkan transkripsi Bahasa Indonesia lewat mikrofon. Audio diproses lokal di memori dan tidak disimpan atau dikirim. Saat pertama kali dijalankan, Whisper mengunduh model `tiny` dari Hugging Face; koneksi internet diperlukan hanya untuk unduhan model awal. Setelah itu model tersimpan di cache komputer.

```powershell
& ".\.venv\Scripts\python.exe" .\comvi\machine_hud.py --known-dir .\comvi\known_faces --mirror --listen
```

Transkrip terbaru tampil di HUD setelah potongan audio sekitar empat detik selesai diproses. `--speech-model base` memilih model yang lebih besar, sedangkan `--speech-device 1` memilih indeks mikrofon tertentu bila perangkat default keliru. Gunakan hanya saat semua orang di sekitar mengetahui mikrofon sedang aktif.

Tekan `Q` atau `Esc` untuk keluar. Sistem ini demo, tidak memiliki pemeriksaan liveness dan bukan autentikasi atau penilaian bahaya.
