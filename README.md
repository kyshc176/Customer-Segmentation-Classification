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

## Cara Menjalankan

1. Jalankan `Clustering_Submission_Akhir_FIXED_Tika_Putri_Marsanti.ipynb` terlebih dahulu — menghasilkan `clustering_data.csv`
2. Jalankan `Klasifikasi_Submission_Akhir_FIXED_Tika_Putri_Marsanti.ipynb` — membaca `clustering_data.csv` dari folder yang sama

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Catatan

Karena label `Cluster` yang diklasifikasikan berasal dari K-Means pada fitur yang sama persis dengan fitur input klasifikasi, akurasi yang sangat tinggi (>99%) merupakan hal yang wajar secara matematis (batas antar kelas relatif tegas berdasarkan jarak ke centroid), bukan semata indikasi overfitting seperti pada kasus klasifikasi dengan label independen/asli.
