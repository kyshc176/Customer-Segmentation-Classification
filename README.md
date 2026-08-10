# Customer Segmentation & Classification — Bank Transactions Data

Proyek machine learning end-to-end yang terdiri dari dua tahap: **Clustering** (unsupervised) untuk melakukan segmentasi nasabah bank berdasarkan pola transaksi, dan **Klasifikasi** (supervised) untuk mempelajari serta memprediksi segmen tersebut dari data baru.

## Deskripsi Proyek

Dataset berisi data transaksi perbankan tanpa label yang mencakup informasi transaksi (jumlah, tipe, durasi, channel), profil nasabah (usia, pekerjaan), dan saldo akun. Karena tidak ada label segmen yang tersedia, proyek ini pertama-tama membangun label segmen sendiri menggunakan **K-Means Clustering**, lalu label hasil clustering tersebut dipakai sebagai target untuk melatih model **klasifikasi** (Random Forest & Decision Tree) — sehingga segmen nasabah baru dapat diprediksi tanpa perlu menjalankan ulang proses clustering dari awal.

## Dataset

| Item | Keterangan |
|---|---|
| Sumber | `bank_transactions_data_2.csv` |
| Ukuran awal | 2.512 baris × 16 kolom |
| Ukuran setelah cleaning | 2.399 baris (113 baris outlier dibuang via metode IQR) |
| Tipe | Data transaksi tanpa label (unsupervised) |
| Fitur utama | TransactionAmount, CustomerAge, TransactionDuration, LoginAttempts, AccountBalance, TransactionType, Channel, CustomerOccupation |

## Struktur File

```
├── Clustering_Submission_Akhir_FIXED_Tika_Putri_Marsanti.ipynb   # Tahap 1: Clustering
├── Klasifikasi_Submission_Akhir_FIXED_Tika_Putri_Marsanti.ipynb  # Tahap 2: Klasifikasi
├── bank_transactions_data_2.csv                                  # Dataset mentah (input clustering)
├── clustering_data.csv                                           # Output clustering (input klasifikasi)
└── README.md
```

## Tahap 1 — Clustering

**Alur:** Perkenalan Dataset → Import Library → Memuat Dataset → EDA → Data Preprocessing → Pembangunan Model → Evaluasi → Feature Selection → Visualisasi → Interpretasi → Export.

**Metodologi:**
- Preprocessing: konversi tanggal, penanganan outlier (IQR), encoding fitur kategorikal (Label Encoding), standardisasi (StandardScaler)
- Fitur clustering: 8 fitur (5 numerik + 3 kategorikal ter-encode)
- Penentuan jumlah cluster optimal: Elbow Method + Silhouette Score → **k = 4**
- Model final di-*fit* satu kali dan dipakai konsisten untuk visualisasi (proyeksi PCA 2D) maupun interpretasi
- Feature selection: dicoba penyaringan fitur berkorelasi tinggi (>0.7) — tidak ada fitur redundan ditemukan, sehingga model baseline (8 fitur) tetap dipakai

**Hasil segmentasi (k=4):**

| Cluster | Jumlah | Profil |
|---|---|---|
| 0 | 402 | Senior Mapan — usia rata-rata 52,7 th, saldo tinggi, transaksi Credit via Branch |
| 1 | 604 | Muda/Pelajar — usia rata-rata 23,2 th, saldo terendah, nominal transaksi relatif tinggi |
| 2 | 1.301 | Profesional Mainstream — segmen terbesar, saldo tinggi, transaksi konservatif via ATM |
| 3 | 92 | Login Attempts Tinggi — rata-rata 3–5 percobaan login (vs ~1 di cluster lain), berpotensi perlu monitoring keamanan akun |

Output: `clustering_data.csv` (dataset asli + kolom `Cluster`).

## Tahap 2 — Klasifikasi

**Alur:** Import Library → Memuat Dataset Hasil Clustering → Data Splitting → Pembangunan Model → Evaluasi → Tuning (GridSearchCV) → Evaluasi Setelah Tuning → Analisis.

**Metodologi:**
- Target: kolom `Cluster` (4 kelas) dari hasil Tahap 1
- Split data latih/uji: 80:20, fitur dinormalisasi dengan StandardScaler
- Model: Random Forest & Decision Tree, dituning dengan GridSearchCV (5-fold CV)

**Hasil evaluasi (data uji, 480 sampel):**

| Model | Akurasi Sebelum Tuning | Akurasi Setelah Tuning |
|---|---|---|
| Random Forest | 99,17% | **99,375%** ↑ |
| Decision Tree | 99,58% | 99,58% (tidak berubah) |

Peningkatan performa Random Forest paling terasa pada Cluster 3 (kelas minoritas, ~3,8% data): recall naik dari 95% menjadi 100% setelah tuning.

## Cara Menjalankan (Notebook)

1. Jalankan `Clustering_Submission_Akhir_FIXED_Tika_Putri_Marsanti.ipynb` terlebih dahulu — menghasilkan `clustering_data.csv`
2. Jalankan `Klasifikasi_Submission_Akhir_FIXED_Tika_Putri_Marsanti.ipynb` — membaca `clustering_data.csv` dari folder yang sama

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Deploy ke Streamlit

Proyek ini juga dilengkapi aplikasi web sederhana (`app.py`) yang memprediksi segmen nasabah dari input transaksi baru secara real-time, memakai model klasifikasi terbaik (Random Forest atau Decision Tree, dipilih otomatis berdasarkan akurasi tertinggi).

**File tambahan untuk deployment:**
- `train_export.py` — menjalankan ulang pipeline preprocessing → clustering → klasifikasi, lalu menyimpan seluruh model/scaler/encoder ke folder `artifacts/` (format `.pkl` via `joblib`) plus `metadata.json` (profil tiap cluster, nama & akurasi model terpilih)
- `app.py` — aplikasi Streamlit yang membaca `artifacts/` dan menyediakan form input untuk memprediksi cluster nasabah baru, lengkap dengan probabilitas prediksi dan perbandingan ke profil rata-rata segmen

**Langkah deploy:**

1. **Generate artifacts (sekali di awal, atau setiap dataset berubah):**
   ```bash
   python train_export.py
   ```
   Ini akan membuat folder `artifacts/` berisi `scaler.pkl`, `clf_scaler.pkl`, `label_encoders.pkl`, `kmeans_model.pkl`, `classifier_model.pkl`, `feature_cols.pkl`, dan `metadata.json`.

2. **Jalankan lokal untuk cek:**
   ```bash
   streamlit run app.py
   ```
   Buka `http://localhost:8501` di browser.

3. **Deploy ke Streamlit Community Cloud (gratis):**
   - Push seluruh folder proyek (termasuk `app.py`, `requirements.txt`, `train_export.py`, `bank_transactions_data_2.csv`) ke repo GitHub — folder `artifacts/` bisa ikut di-push, atau dibuat otomatis saat startup dengan menambahkan pemanggilan `train_export.py` di awal `app.py`
   - Buka [share.streamlit.io](https://share.streamlit.io), hubungkan ke repo GitHub tersebut
   - Pilih `app.py` sebagai entry point, deploy
   - Streamlit Cloud otomatis membaca `requirements.txt` untuk install dependencies

4. **Alternatif deploy lain:** Hugging Face Spaces (pilih SDK "Streamlit"), atau Docker + Railway/Render kalau butuh kontrol lebih (tinggal tambahkan `Dockerfile` sederhana dengan base image `python:3.11-slim` + `pip install -r requirements.txt` + `CMD ["streamlit","run","app.py"]`).

**Fitur app.py:**
- Form input 8 fitur transaksi (jumlah, usia, durasi, login attempts, saldo, tipe transaksi, channel, pekerjaan)
- Prediksi segmen (Cluster 0–3) beserta nama deskriptifnya
- Grafik probabilitas prediksi per cluster
- Tabel perbandingan ke karakteristik rata-rata segmen tersebut

## Catatan

Karena label `Cluster` yang diklasifikasikan berasal dari K-Means pada fitur yang sama persis dengan fitur input klasifikasi, akurasi yang sangat tinggi (>99%) merupakan hal yang wajar secara matematis (batas antar kelas relatif tegas berdasarkan jarak ke centroid), bukan semata indikasi overfitting seperti pada kasus klasifikasi dengan label independen/asli.
