# Laporan UAS Proyek Deep Learning
# Sistem Deteksi dan Pembacaan Plat Nomor Kendaraan (YOLOv8 + PaddleOCR)

## 1. Pendahuluan

Dokumen ini adalah **makalah rangkuman UAS Proyek Deep Learning** yang merangkum seluruh artefak evaluasi UAS (proyek, paper ilmiah, source code, dan deployment) sesuai ketentuan.

---

## 2. Deskripsi Proyek UAS Deep Learning

Proyek UAS ini berfokus pada **pengembangan dan analisis sistem Deep Learning** untuk kebutuhan nyata: **deteksi dan pembacaan plat nomor kendaraan** dari foto. Output sistem adalah JSON berisi teks plat (mentah dan terformat) serta (opsional) **lookup wilayah Samsat** berdasarkan plat.

Ruang lingkup UAS yang dipenuhi:

- **Fokus UAS pada Deep Learning**: model utama adalah **object detection** (YOLOv8) dan OCR berbasis DL (PaddleOCR) untuk membaca teks pada plat.
- **Proyek berdampak nyata**: dapat digunakan untuk otomasi validasi/identifikasi plat pada monitoring kendaraan.
- **Keterkaitan proyek–paper–deployment**:
  - Proyek mengadopsi metode ALPR/ANPR modern (deteksi → crop → OCR).
  - Implementasi disajikan sebagai backend API (Go) yang memanggil worker inferensi (Python).
  - Disediakan langkah deployment (VM + reverse proxy Nginx + service) untuk demo layanan.

---

## 3. Ketentuan dan Ruang Lingkup Proyek

### 3.1 Model Deep Learning yang digunakan

- **YOLOv8 (Ultralytics)**: single-stage CNN object detector untuk **mendeteksi bounding box plat**.
  - Fine-tuning dari model dasar `yolov8n.pt` menjadi model custom `best.pt`.
- **PaddleOCR (PaddlePaddle)**: OCR berbasis DL untuk **mengenali karakter plat** dari hasil crop.
  - Konfigurasi runtime: `use_angle_cls=True`, `use_gpu=False`.

Catatan: pipeline ini merupakan **kombinasi model deteksi + OCR** (hybrid pipeline), bukan RNN murni (LSTM/GRU) atau Transformer murni. OCR dilakukan oleh model PaddleOCR bawaan library.

### 3.2 Kompleksitas masalah

Masalah termasuk kompleks karena melibatkan:

- Variasi kondisi foto: sudut, jarak, blur, noise, pencahayaan, background kompleks.
- Deteksi objek kecil (plat) dan ketergantungan tahap berikutnya: kegagalan deteksi akan menggagalkan OCR.
- OCR pada teks pendek yang sensitif terhadap karakter mirip (mis. `O↔0`, `I↔1`, `B↔8`).

### 3.3 Dataset yang digunakan dan preprocessing

**Dataset**: Roboflow Universe — *Plat nomor kendaraan Indonesia*  
Link: `https://universe.roboflow.com/kuraimos/plat-nomor-kendaraan-indonesia-dowl5`

Informasi dataset:

- Total **1409 images**
- Anotasi: **YOLOv8 format**
- Kelas: **1 kelas** (`plat_nomor`)
- Split:
  - `train`: `Plat-nomor-kendaraan-Indonesia-1/train/images`
  - `val`: `Plat-nomor-kendaraan-Indonesia-1/valid/images`
  - `test`: `Plat-nomor-kendaraan-Indonesia-1/test/images`
- Preprocessing Roboflow:
  - auto-orient (EXIF stripped)
  - resize ke **640×640 (stretch)**
- Augmentasi (3 versi per source image):
  - horizontal flip (50%)
  - brightness (-10% s/d +10%)
  - Gaussian blur (0–1 px)
  - salt & pepper noise (0.1% pixel)

**Preprocessing sebelum OCR (runtime inference, sesuai `backend-python/detect.py` dan `paddleOCR.py`)**:

1. Crop area plat dari hasil deteksi YOLO + padding agar karakter tepi tidak terpotong.
2. Resize crop (pembesaran) untuk membantu OCR.
3. Grayscale → Gaussian blur → **Otsu thresholding** untuk meningkatkan kontras karakter.

Rujukan Otsu threshold:
- Otsu, 1979 — DOI `10.1109/tsmc.1979.4310076`

### 3.4 Evaluasi performa model

Evaluasi deteksi plat dilakukan memakai metrik object detection dari Ultralytics (val):

- Precision (B): **0.9831**
- Recall (B): **0.9917**
- mAP@0.50 (B): **0.9947**
- mAP@0.50:0.95 (B): **0.9341**

Catatan:

- Metrik diambil dari hasil training YOLOv8 pada 30 epoch (`runs/detect/train/results.csv`).
- Konfigurasi training utama:
  - Model base: `yolov8n.pt`
  - Epoch: 30
  - Image size: 640
  - Batch: 8
  - Device: GPU jika tersedia (`cuda: True` pada notebook).
- Log inferensi contoh (dari notebook):
  - `preprocess 4.6ms` + `inference 12.7ms` + `postprocess 1.5ms` pada sample image.

---

## 4. Paper Penelitian Ilmiah

### 4.1 Deskripsi Paper

Paper untuk UAS:

- Ditulis dalam **Bahasa Inggris**
- Format **Word (`.docx`)**
- Mengikuti standar jurnal/konferensi
  - RESTI: `https://jurnal.iaii.or.id/index.php/RESTI`
  - IJAIDM: `https://ejournal.uin-suska.ac.id/index.php/IJAIDM/index`
  - ICIC (IEEE template): `https://icic-aptikom.org/2025/`

**Paper acuan yang sesuai metode proyek (YOLOv8 + PaddleOCR)**:

- *Design of Vehicle License Plate Detection System Using YOLOv8 and PaddleOCR* — DOI `10.1109/icera66156.2025.11087294`
- *NLPDRS: A YOLOv8 and PaddleOCR-Based End-to-End Framework for Nigerian License Plate Detection and Recognition System* — DOI `10.26438/ijsrcse.v13i5.748`
- *License Plate Detection using YOLO v8 and Performance Evaluation of EasyOCR, PaddleOCR and Tesseract* — DOI `10.1109/icccnt61001.2024.10725878`
- (Pendukung deteksi) *You Only Look Once: Unified, Real-Time Object Detection* — DOI `10.1109/cvpr.2016.91`

### 4.2 Struktur Paper

- Introduction
- Related Work
- Methodology (Deep Learning Architecture)
- Experiments and Results
- Discussion
- Conclusion

### 4.3 Link Paper

Link Paper (Word):

- Link tidak tersedia pada repositori lokal.

---

## 5. Laporan Teknis Proyek Deep Learning

Bagian ini memetakan kriteria penilaian teknis:

- **Kompleksitas masalah**: deteksi + OCR berantai (pipeline) pada kondisi foto unconstrained.
- **Dataset**: Roboflow Universe plat nomor Indonesia (YOLOv8 format) + augmentasi.
- **Preprocessing**:
  - YOLO: mengikuti preprocessing dataset (resize 640×640).
  - OCR: crop + resize + grayscale + blur + Otsu threshold.
- **Arsitektur model**:
  - Deteksi: YOLOv8n fine-tuned (Ultralytics).
  - OCR: PaddleOCR (angle classifier aktif).
- **Minimal 2 fitur unik yang ditambahkan**:
  1. **Normalisasi plat Indonesia** dari hasil OCR (heuristik perbaikan karakter + regex pola plat) sehingga output konsisten (contoh: `BK 4272 AMQ`).
  2. **Lookup wilayah Samsat** berdasarkan plat (mengambil referensi dari `samsat.info`) sebagai nilai tambah berbasis kebutuhan nyata.
- **Hasil evaluasi (akurasi/error)**:
  - Deteksi (YOLO): precision/recall/mAP tercantum pada Bagian 3.4.
  - OCR: evaluasi bersifat kualitatif pada sampel uji; disarankan menambah metrik CER/WER pada test set untuk pelaporan final.
- **Teknologi dan deployment**:
  - Backend API: Go (docs + endpoint upload).
  - Worker inferensi: Python (YOLO + OCR) dipanggil oleh Go via `os/exec`.
  - Deployment: VM + Nginx reverse proxy + systemd service (opsional).

Link Laporan Teknis (PDF):

- Link tidak tersedia pada repositori lokal.

---

## 6. Source Code dan Repositori

### 6.1 Platform Repositori

Repositori belum dipublikasikan ke platform eksternal pada saat laporan dibuat.

### 6.2 Struktur Source Code

**Folder Deep Learning (training + eksperimen):**

- `E:\RPL\Semester 7\ML\DeepL\UAS_MachineLearning.ipynb`  
  Notebook untuk setup, training, validasi, dan inferensi YOLOv8.
- `E:\RPL\Semester 7\ML\DeepL\paddleOCR.py`  
  Skrip inferensi YOLOv8 + preprocessing + PaddleOCR (standalone).
- `E:\RPL\Semester 7\ML\DeepL\Plat-nomor-kendaraan-Indonesia-1\`  
  Dataset lengkap (train/valid/test + labels).
- `E:\RPL\Semester 7\ML\DeepL\runs\detect\train\`  
  Hasil training YOLOv8 (weights, metrics, grafik).

**Folder UAS (backend aplikasi):**

- `E:\RPL\Semester 7\ML\UAS\backend-python\detect.py`  
  Worker inferensi: YOLOv8 + PaddleOCR + preprocessing + output JSON.
- `E:\RPL\Semester 7\ML\UAS\backend-python\utils\ocr_cleaner.py`  
  Normalisasi teks OCR untuk format plat Indonesia.
- `E:\RPL\Semester 7\ML\UAS\backend-go\`  
  REST API: upload → panggil Python → return JSON + (opsional) Samsat lookup.
- Dokumentasi: `E:\RPL\Semester 7\ML\UAS\README.md`, `backend-go/README.md`, `backend-python/README.md`

Link Source Code:

- Link tidak tersedia pada repositori lokal.

---

## 7. Implementasi, Demo Program, dan Deployment

### 7.1 Bentuk Aplikasi

- **Web API (Backend Service)**: endpoint `POST /detect` menerima upload foto, mengembalikan JSON hasil deteksi + OCR.

### 7.2 Arsitektur Sistem

**Alur utama:**

1. Client upload gambar ke `POST /detect` (multipart/form-data).
2. Go server menyimpan file sementara.
3. Go memanggil Python worker (`detect.py`) dengan path file.
4. Python melakukan YOLO → crop → preprocess → OCR → normalisasi.
5. Hasil JSON dikembalikan ke Go.
6. (Opsional) Go melakukan lookup Samsat via `samsat.info` (Firestore).
7. Response final dikirim ke client.

**Komponen:**

- **Backend Go**:
  - Router + middleware (logging, request id, recovery).
  - Endpoint: `GET /healthz`, `POST /detect`, `GET /openapi.json`, docs UI (`/`).
- **Worker Python**:
  - YOLOv8 custom (`best.pt`).
  - PaddleOCR (CPU).
  - Output JSON terstruktur.
- **Samsat lookup (opsional)**:
  - Scraper via Firestore API (`samsat.info`).

### 7.3 Konfigurasi Environment

**Python dependencies** (ringkas, lihat `backend-python/requirements.txt`):

- `ultralytics`
- `opencv-python`
- `paddlepaddle`
- `paddleocr`
- `numpy`
- `Pillow`

**Go requirements:**

- Go >= 1.22

**Environment variables utama (Go):**

- `YOLO_PY_SCRIPT` (wajib): path ke `backend-python/detect.py`
- `PYTHON_BIN` (default `python`)
- `ADDR` (default `:8080`)
- `YOLO_TIMEOUT_SECONDS` (default `120`)
- `SCRAPE_TIMEOUT_SECONDS` (default `15`)
- `MAX_UPLOAD_MB` (default `15`)
- `MIN_PLATE_CONFIDENCE` (default `0`)
- `SAMSAT_FIRESTORE_API_KEY`, `SAMSAT_PAGE_URL`, `SAMSAT_FIRESTORE_BASE_URL`

**Hardware (ringkas):**

- CPU: tidak terdokumentasi pada repositori lokal.
- GPU: tidak terdokumentasi pada repositori lokal.
- RAM: tidak terdokumentasi pada repositori lokal.

### 7.4 Analisis Performa

**Waktu inferensi:**

- Contoh log YOLO pada notebook: ~**12.7 ms** inference per image (di GPU).
- Latensi end-to-end dipengaruhi oleh:
  - ukuran gambar,
  - proses OCR (CPU),
  - overhead pemanggilan Python dari Go.

**Resource usage:**

- YOLO inference menambah penggunaan GPU/CPU.
- OCR menambah penggunaan CPU/RAM.
- Proses Python di-cache dalam runtime worker (model tidak di-load ulang jika proses tetap hidup).

---

## Lampiran: Ringkasan Implementasi (Detail Teknis)

### A. Pipeline YOLOv8 Training (Notebook)

- Setup dataset lokal (`data.yaml`).
- Training: `YOLO('yolov8n.pt')`, `epochs=30`, `imgsz=640`, `batch=8`, `device=GPU/CPU`.
- Validasi: `model.val(data=DATA_YAML)`.
- Hasil weights: `runs/detect/train/weights/best.pt`.

### B. Pipeline Inference + OCR (Python)

Langkah utama dalam `backend-python/detect.py`:

1. Load YOLO model (`best.pt`).
2. Detect plat dengan confidence `0.25`.
3. Crop plat + padding.
4. Resize crop (pembesaran).
5. Grayscale → blur → Otsu threshold.
6. Jalankan PaddleOCR pada:
   - citra crop asli
   - citra threshold
7. Pilih hasil OCR terbaik.
8. Normalisasi teks plat (regex + heuristik karakter).
9. Output JSON: `plate_raw`, `plate_cleaned`, `confidence`.

### C. Normalisasi Teks Plat

Heuristik perbaikan OCR (`ocr_cleaner.py`):

- `O→0`, `I/L→1`, `Z→2`, `S→5`, `B→8`, `G→6`
- Regex pola plat Indonesia: `([A-Z]{1,2})(\\d{1,4})([A-Z]{1,3})`
- Output final dibatasi maksimal 9 karakter jika tidak match pola.
