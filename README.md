# Laporan UAS Proyek Deep Learning — Plate Detection System (YOLOv8 + PaddleOCR)

## 1. Pendahuluan

Dokumen ini adalah **makalah rangkuman UAS Proyek Deep Learning** yang merangkum seluruh artefak evaluasi UAS (proyek, paper ilmiah, source code, dan deployment) sesuai ketentuan.

---

## 2. Deskripsi Proyek UAS Deep Learning

Proyek UAS ini berfokus pada **pengembangan dan analisis sistem Deep Learning** untuk kebutuhan nyata: **deteksi dan pembacaan plat nomor kendaraan** dari foto, kemudian (opsional) melakukan **lookup wilayah Samsat** berdasarkan plat.

Ruang lingkup UAS yang dipenuhi:

- **Fokus UAS pada Deep Learning**: model utama adalah **object detection** (YOLOv8) dan OCR berbasis DL (PaddleOCR) untuk membaca teks pada plat.
- **Proyek berdampak nyata**: output sistem dapat dipakai untuk otomasi validasi/identifikasi plat pada konteks monitoring kendaraan.
- **Keterkaitan proyek–paper–deployment**:
  - Proyek mengadopsi metode yang sesuai paper ALPR/ANPR modern (deteksi → crop → OCR).
  - Hasil implementasi disajikan sebagai aplikasi backend (Go) yang memanggil worker inferensi (Python).
  - Disediakan langkah deployment (VM + reverse proxy Nginx) agar sistem bisa didemokan sebagai layanan.

---

## 3. Ketentuan dan Ruang Lingkup Proyek

### 3.1 Model Deep Learning yang digunakan

- **YOLOv8 (Ultralytics)**: CNN-based object detector (single-stage) untuk **mendeteksi bounding box plat**.
  - Fine-tuning dari model dasar `yolov8n.pt` menjadi model custom `best.pt`.
- **PaddleOCR (PaddlePaddle)**: OCR untuk **mengenali karakter plat** dari hasil crop.
  - OCR dijalankan dengan `use_angle_cls=True` dan `use_gpu=False` (stabil untuk runtime worker).

### 3.2 Kompleksitas masalah

Masalah termasuk kompleks karena melibatkan:

- Variasi kondisi foto: sudut, jarak, blur, noise, pencahayaan, background kompleks.
- Deteksi objek kecil (plat) dan ketergantungan tahap berikutnya: kegagalan deteksi akan menggagalkan OCR.
- OCR pada teks pendek yang sensitif terhadap karakter mirip (mis. `O↔0`, `I↔1`, `B↔8`).

### 3.3 Dataset yang digunakan dan preprocessing

**Dataset**: Roboflow Universe — *Plat nomor kendaraan Indonesia*  
Link: `https://universe.roboflow.com/kuraimos/plat-nomor-kendaraan-indonesia-dowl5`

Informasi penting dataset:

- Total **1409 images**
- Anotasi: **YOLOv8 format**
- Preprocessing Roboflow:
  - auto-orient (EXIF stripped)
  - resize ke **640×640 (stretch)**
- Augmentasi (3 versi per source image):
  - horizontal flip (50%)
  - brightness (-10% s/d +10%)
  - Gaussian blur (0–1 px)
  - salt & pepper noise (0.1% pixel)

**Preprocessing sebelum OCR (runtime inference)**:

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

Catatan: metrik diambil dari hasil training YOLOv8 pada 30 epoch (`runs/detect/train/results.csv`) dengan konfigurasi umum:

- Model base: `yolov8n.pt`
- Epoch: 30
- Image size: 640
- Batch: 8

---

## 4. Paper Penelitian Ilmiah

### 4.1 Deskripsi Paper

Paper yang dibuat untuk UAS:

- Ditulis dalam **Bahasa Inggris**
- Format **Word (`.docx`)**
- Mengikuti standar jurnal/konferensi (contoh template):
  - RESTI: `https://jurnal.iaii.or.id/index.php/RESTI`
  - IJAIDM: `https://ejournal.uin-suska.ac.id/index.php/IJAIDM/index`
  - ICIC (sering menggunakan template IEEE): `https://icic-aptikom.org/2025/`

Paper acuan yang sesuai **metode yang benar-benar dipakai di proyek (YOLOv8 + PaddleOCR)**:

- *Design of Vehicle License Plate Detection System Using YOLOv8 and PaddleOCR* — DOI `10.1109/icera66156.2025.11087294`
- *NLPDRS: A YOLOv8 and PaddleOCR-Based End-to-End Framework for Nigerian License Plate Detection and Recognition System* — DOI `10.26438/ijsrcse.v13i5.748`
- (Pendukung deteksi) *You Only Look Once: Unified, Real-Time Object Detection* — DOI `10.1109/cvpr.2016.91`

### 4.2 Struktur Paper

Struktur paper yang digunakan:

- Introduction
- Related Work
- Methodology (Deep Learning Architecture)
- Experiments and Results
- Discussion
- Conclusion

### 4.3 Link Paper

Link Paper (Word):

- (isi link di sini) `https://...`

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
  - OCR: PaddleOCR (dengan angle classifier).
- **Minimal 2 fitur unik yang ditambahkan**:
  1. **Normalisasi plat Indonesia** dari hasil OCR (heuristik perbaikan karakter + regex pola plat) sehingga output konsisten (contoh: `BK4272AMQ`).
  2. **Lookup wilayah Samsat** berdasarkan plat (mengambil referensi dari `samsat.info`) sebagai nilai tambah berbasis kebutuhan nyata.
- **Hasil evaluasi (akurasi/error)**:
  - Deteksi (YOLO): precision/recall/mAP tercantum pada Bagian 3.4.
  - OCR: dievaluasi secara kualitatif pada sampel uji (disarankan menambah pengukuran CER/WER pada dataset test untuk laporan final).
- **Teknologi dan deployment**:
  - Backend API: Go (serve docs + endpoint upload).
  - Worker inferensi: Python (YOLO + OCR) dipanggil oleh Go via `os/exec`.
  - Deployment: VM + Nginx reverse proxy + systemd service (opsional).

Link Laporan Teknis (PDF):

- (isi link di sini) `https://...`

---

## 6. Source Code dan Repositori

### 6.1 Platform Repositori

- GitHub / OneDrive: (pilih salah satu)

### 6.2 Struktur Source Code

- **Preprocessing**: dilakukan saat inference (crop, resize, threshold) di `backend-python/detect.py`.
- **Model Deep Learning**:
  - YOLO weights: `backend-python/model/best.pt`
  - OCR: PaddleOCR (runtime)
- **Training & Evaluation**: notebook/training log tersimpan pada folder training YOLO (mis. `runs/detect/train/`).
- **Dokumentasi**: `README.md` (root), `backend-go/README.md`, `backend-python/README.md`

Link Source Code:

- (isi link di sini) `https://...`

---

## 7. Implementasi, Demo Program, dan Deployment

### 7.1 Bentuk Aplikasi

- **Web API (Backend Service)**: endpoint `POST /detect` menerima upload foto, mengembalikan JSON hasil deteksi + OCR.

### 7.2 Arsitektur Sistem

- **Backend**: Go (`backend-go/`) menyediakan REST API + halaman docs.
- **Worker inferensi**: Python (`backend-python/`) menjalankan YOLOv8 + PaddleOCR dan print hasil JSON ke stdout.
- **Model deployment**: model disimpan lokal pada server (`backend-python/model/best.pt`) dan dipanggil per request.

### 7.3 Konfigurasi Environment

Library utama (ringkas):

- Python: `ultralytics`, `opencv-python`, `paddlepaddle`, `paddleocr`, `numpy`
- Go: Go >= 1.22 (server API)

Hardware (diisi sesuai perangkat yang dipakai):

- CPU: (isi)
- GPU (opsional): (isi)
- RAM: (isi)

### 7.4 Analisis Performa

- **Waktu inferensi**:
  - Dipengaruhi oleh ukuran gambar, device (CPU/GPU), dan latensi load model (cache model pada proses worker membantu).
  - (isi hasil pengukuran: ms per request pada perangkat uji)
- **Resource usage**:
  - YOLO inference dan OCR menambah penggunaan CPU/RAM; jika memakai GPU untuk YOLO, VRAM akan terpakai.
  - (isi observasi penggunaan resource saat demo)

