"""
app.py — Dashboard Prediksi Kelompok Nasabah Bank

Dashboard multi-halaman:
1. Overview & EDA
2. Prediksi Individual
3. Prediksi Batch (CSV)
4. Model Insight

Model:
- Label Encoding untuk fitur kategorikal
- StandardScaler
- Classifier hasil training
- Label/segment berasal dari clustering K-Means

Struktur folder yang dibutuhkan:

project/
│
├── app.py
├── bank_transactions_data_2.csv
│
└── artifacts/
    ├── clf_scaler.pkl
    ├── label_encoders.pkl
    ├── classifier_model.pkl
    ├── feature_cols.pkl
    └── metadata.json

Jalankan:
    streamlit run app.py
"""

# ============================================================
# 1. IMPORT
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# 2. KONFIGURASI PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "bank_transactions_data_2.csv"

SCALER_PATH = ARTIFACTS_DIR / "clf_scaler.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "label_encoders.pkl"
MODEL_PATH = ARTIFACTS_DIR / "classifier_model.pkl"
FEATURE_PATH = ARTIFACTS_DIR / "feature_cols.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


# ============================================================
# 3. KONFIGURASI STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Dashboard Segmentasi Nasabah Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 4. CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    /* Metric */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 10px;
        padding: 12px;
        background: rgba(128, 128, 128, 0.04);
    }

    /* Result card */
    .prediction-card {
        border-radius: 14px;
        padding: 22px;
        margin: 10px 0 20px 0;
        border: 1px solid rgba(128, 128, 128, 0.20);
    }

    .prediction-emoji {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .prediction-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .prediction-description {
        font-size: 15px;
        color: #666;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        padding-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. INFORMASI SEGMENT
# ============================================================

SEGMENT_INFO = {
    0: {
        "nama": "Nasabah Senior Mapan",
        "emoji": "🧓",
        "warna": "#0F6E56",
        "ringkas": (
            "Nasabah usia lebih dewasa dengan kondisi keuangan yang stabil."
        ),
        "detail": (
            "Kelompok ini biasanya nasabah berusia lebih tua, "
            "sudah tidak lagi aktif bekerja, dengan saldo tabungan "
            "yang cukup besar. Mereka lebih suka datang langsung "
            "ke kantor cabang."
        ),
        "saran": (
            "Cocok ditawari produk deposito, investasi jangka panjang, "
            "atau layanan prioritas di cabang."
        ),
    },
    1: {
        "nama": "Nasabah Muda / Pelajar",
        "emoji": "🎓",
        "warna": "#185FA5",
        "ringkas": (
            "Nasabah usia muda, kemungkinan besar masih berstatus "
            "pelajar atau mahasiswa."
        ),
        "detail": (
            "Saldo tabungan kelompok ini relatif kecil, tetapi "
            "mereka cukup aktif bertransaksi dibandingkan jumlah "
            "saldo yang dimiliki."
        ),
        "saran": (
            "Cocok ditawari tabungan pelajar, promo cashback, "
            "atau edukasi pengelolaan keuangan."
        ),
    },
    2: {
        "nama": "Nasabah Profesional",
        "emoji": "💼",
        "warna": "#534AB7",
        "ringkas": (
            "Kelompok terbesar — nasabah dengan pekerjaan tetap "
            "dan kondisi keuangan mapan."
        ),
        "detail": (
            "Kelompok ini mencakup nasabah usia menengah ke atas "
            "yang sudah bekerja tetap, dengan saldo tinggi namun "
            "transaksi cenderung rutin dan tidak besar-besar. "
            "Mereka lebih suka bertransaksi lewat ATM."
        ),
        "saran": (
            "Cocok ditawari kartu kredit, cicilan, "
            "atau produk investasi menengah."
        ),
    },
    3: {
        "nama": "Perlu Perhatian Keamanan",
        "emoji": "⚠️",
        "warna": "#A32D2D",
        "ringkas": (
            "Nasabah dengan pola percobaan login yang jauh lebih "
            "sering dari biasanya."
        ),
        "detail": (
            "Kelompok kecil ini memiliki satu ciri khas: "
            "mereka mencoba login berkali-kali sebelum berhasil "
            "masuk ke akun. Rata-rata sekitar 3–5 kali, dibandingkan "
            "kelompok lain yang biasanya langsung berhasil."
        ),
        "saran": (
            "Sebaiknya dipantau lebih lanjut — bisa jadi nasabah "
            "lupa kata sandi atau terdapat indikasi percobaan "
            "akses tidak sah."
        ),
    },
}


# ============================================================
# 6. KOLOM YANG DIGUNAKAN
# ============================================================

CATEGORICAL_COLUMNS = [
    "TransactionType",
    "Channel",
    "CustomerOccupation",
]

REQUIRED_BATCH_COLUMNS = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "TransactionType",
    "Channel",
    "CustomerOccupation",
]

NUMERIC_COLUMNS = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
]


# ============================================================
# 7. LOAD MODEL / ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    """
    Membaca seluruh artifact model.
    """

    required_files = [
        SCALER_PATH,
        ENCODER_PATH,
        MODEL_PATH,
        FEATURE_PATH,
        METADATA_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Artifact berikut tidak ditemukan:\n"
            + "\n".join(missing_files)
        )

    clf_scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    classifier = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return (
        clf_scaler,
        label_encoders,
        classifier,
        feature_cols,
        metadata,
    )


# ============================================================
# 8. LOAD DATA
# ============================================================

@st.cache_data
def load_raw_data():
    """
    Membaca dataset utama.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# 9. ENCODING DATA
# ============================================================

def encode_row(df_raw, label_encoders):
    """
    Mengubah fitur kategorikal menjadi fitur encoded
    berdasarkan LabelEncoder yang sudah digunakan saat training.

    Jika kategori tidak dikenal model, nilainya menjadi NaN.
    """

    df = df_raw.copy()

    for col in CATEGORICAL_COLUMNS:

        if col not in df.columns:
            continue

        if col not in label_encoders:
            raise KeyError(
                f"Label encoder untuk kolom '{col}' tidak ditemukan."
            )

        le = label_encoders[col]

        known_values = set(le.classes_)

        df[f"{col}_encoded"] = df[col].apply(
            lambda value: (
                le.transform([value])[0]
                if value in known_values
                else np.nan
            )
        )

    return df


# ============================================================
# 10. HELPER: SEGMENT LABEL
# ============================================================

def get_segment_info(segment_id):
    """
    Mengambil informasi segment berdasarkan ID.
    """

    return SEGMENT_INFO.get(
        int(segment_id),
        {
            "nama": f"Kelompok {segment_id}",
            "emoji": "❔",
            "warna": "#888888",
            "ringkas": "",
            "detail": "",
            "saran": "",
        },
    )


def segment_label(segment_id):
    """
    Mengubah ID segment menjadi label yang mudah dibaca.
    """

    info = get_segment_info(segment_id)

    return f"{info['emoji']} {info['nama']}"


# ============================================================
# 11. HELPER: VALIDASI FITUR
# ============================================================

def validate_feature_columns(feature_cols):
    """
    Memastikan feature_cols dari artifact tersedia
    dalam format yang diharapkan.
    """

    if not isinstance(feature_cols, (list, tuple)):
        raise ValueError(
            "feature_cols.pkl harus berupa list atau tuple."
        )

    feature_cols = list(feature_cols)

    return feature_cols


# ============================================================
# 12. LOAD STATUS
# ============================================================

artifacts_loaded = True
artifacts_error = None

try:
    (
        clf_scaler,
        label_encoders,
        classifier,
        feature_cols,
        metadata,
    ) = load_artifacts()

    feature_cols = validate_feature_columns(feature_cols)

except Exception as e:
    artifacts_loaded = False
    artifacts_error = str(e)


data_loaded = True
data_error = None

try:
    raw_df = load_raw_data()

except Exception as e:
    data_loaded = False
    data_error = str(e)


# ============================================================
# 13. SIDEBAR
# ============================================================

st.sidebar.title("🏦 Dashboard Segmentasi Nasabah")

page = st.sidebar.radio(
    "Navigasi",
    [
        "📊 Overview & EDA",
        "🔍 Prediksi Individual",
        "📁 Prediksi Batch (CSV)",
        "📈 Model Insight",
    ],
)


st.sidebar.divider()


if artifacts_loaded:

    best_model_name = metadata.get(
        "best_model_name",
        "Tidak tersedia",
    )

    best_model_accuracy = metadata.get(
        "best_model_accuracy",
        0,
    )

    st.sidebar.caption(
        f"**Model:** {best_model_name}"
    )

    st.sidebar.caption(
        f"**Akurasi:** {best_model_accuracy * 100:.2f}%"
    )

    st.sidebar.caption(
        "Segmen: K-Means (k=4)"
    )

else:

    st.sidebar.error(
        "Artifact model belum tersedia."
    )


st.sidebar.divider()

st.sidebar.caption(
    "Dashboard Prediksi Kelompok Nasabah"
)

st.sidebar.caption(
    "Bank Transaction Analytics"
)


# ============================================================
# ============================================================
# HALAMAN 1 — OVERVIEW & EDA
# ============================================================
# ============================================================

if page == "📊 Overview & EDA":

    st.title("📊 Overview & Exploratory Data Analysis")

    st.markdown(
        """
        Dashboard ini digunakan untuk melihat distribusi data transaksi
        serta karakteristik masing-masing kelompok nasabah hasil segmentasi.
        """
    )

    # --------------------------------------------------------
    # Validasi
    # --------------------------------------------------------

    if not data_loaded:

        st.error(
            "⚠️ Dataset `bank_transactions_data_2.csv` tidak ditemukan."
        )

        st.code(
            str(DATA_PATH),
            language="text",
        )

        st.stop()

    if not artifacts_loaded:

        st.error(
            "⚠️ Artifact model tidak lengkap."
        )

        st.code(
            artifacts_error,
            language="text",
        )

        st.info(
            "Pastikan folder `artifacts/` berisi file hasil training."
        )

        st.stop()

    # --------------------------------------------------------
    # Validasi data
    # --------------------------------------------------------

    missing_required = [
        col
        for col in REQUIRED_BATCH_COLUMNS
        if col not in raw_df.columns
    ]

    if missing_required:

        st.error(
            "Dataset tidak memiliki kolom yang dibutuhkan:"
        )

        st.write(missing_required)

        st.stop()

    # --------------------------------------------------------
    # Prediction seluruh dataset
    # --------------------------------------------------------

    with st.spinner(
        "Menghitung segmentasi seluruh nasabah..."
    ):

        df_encoded = encode_row(
            raw_df,
            label_encoders,
        )

        encoded_features = [
            col
            for col in feature_cols
            if col.endswith("_encoded")
        ]

        missing_features = [
            col
            for col in feature_cols
            if col not in df_encoded.columns
        ]

        if missing_features:

            st.error(
                "Fitur berikut tidak tersedia setelah preprocessing:"
            )

            st.write(missing_features)

            st.stop()

        valid_mask = (
            ~df_encoded[feature_cols]
            .isna()
            .any(axis=1)
        )

        valid = df_encoded.loc[
            valid_mask
        ].copy()

        if valid.empty:

            st.error(
                "Tidak ada baris valid yang dapat diprediksi."
            )

            st.stop()

        X_all = valid[feature_cols]

        X_scaled = clf_scaler.transform(
            X_all
        )

        valid["Segment"] = (
            classifier
            .predict(X_scaled)
            .astype(int)
        )

        valid["Segment_Nama"] = (
            valid["Segment"]
            .apply(segment_label)
        )

    # --------------------------------------------------------
    # Informasi data
    # --------------------------------------------------------

    st.subheader("📌 Ringkasan Dataset")

    total_original = len(raw_df)
    total_valid = len(valid)
    total_invalid = total_original - total_valid

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Total Data",
        f"{total_original:,}",
    )

    d2.metric(
        "Data Berhasil Diprediksi",
        f"{total_valid:,}",
    )

    d3.metric(
        "Data Tidak Dikenali",
        f"{total_invalid:,}",
    )

    d4.metric(
        "Jumlah Segmen",
        f"{valid['Segment'].nunique()}",
    )

    st.divider()

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    with st.expander(
        "🔧 Filter Data",
        expanded=False,
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            segment_options = sorted(
                valid["Segment_Nama"].dropna().unique()
            )

            seg_filter = st.multiselect(
                "Segmen",
                options=segment_options,
            )

        with c2:

            occupation_options = sorted(
                valid["CustomerOccupation"]
                .dropna()
                .unique()
            )

            occ_filter = st.multiselect(
                "Pekerjaan",
                options=occupation_options,
            )

        with c3:

            min_age = int(
                valid["CustomerAge"].min()
            )

            max_age = int(
                valid["CustomerAge"].max()
            )

            age_range = st.slider(
                "Rentang Usia",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age),
            )

    filtered = valid.copy()

    if seg_filter:

        filtered = filtered[
            filtered["Segment_Nama"].isin(
                seg_filter
            )
        ]

    if occ_filter:

        filtered = filtered[
            filtered["CustomerOccupation"].isin(
                occ_filter
            )
        ]

    filtered = filtered[
        (
            filtered["CustomerAge"]
            >= age_range[0]
        )
        &
        (
            filtered["CustomerAge"]
            <= age_range[1]
        )
    ]

    # --------------------------------------------------------
    # Metrik
    # --------------------------------------------------------

    st.subheader("📊 Ringkasan Setelah Filter")

    m1, m2, m3, m4 = st.columns(4)

    if filtered.empty:

        m1.metric("Total Nasabah", "0")
        m2.metric("Rata-rata Usia", "-")
        m3.metric("Rata-rata Saldo", "-")
        m4.metric("Rata-rata Login", "-")

        st.warning(
            "Tidak ada data yang sesuai dengan filter."
        )

        st.stop()

    else:

        m1.metric(
            "Total Nasabah",
            f"{len(filtered):,}",
        )

        m2.metric(
            "Rata-rata Usia",
            f"{filtered['CustomerAge'].mean():.0f} th",
        )

        m3.metric(
            "Rata-rata Saldo",
            f"{filtered['AccountBalance'].mean():,.0f}",
        )

        m4.metric(
            "Rata-rata Percobaan Login",
            f"{filtered['LoginAttempts'].mean():.1f}",
        )

    st.divider()

    # ========================================================
    # CHART 1 — DISTRIBUSI SEGMENT
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        seg_count = (
            filtered["Segment_Nama"]
            .value_counts()
            .reset_index()
        )

        seg_count.columns = [
            "Segmen",
            "Jumlah",
        ]

        fig = px.pie(
            seg_count,
            names="Segmen",
            values="Jumlah",
            title="Distribusi Segmen Nasabah",
            hole=0.35,
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # CHART 2 — SALDO PER SEGMENT
    # ========================================================

    with col2:

        fig = px.box(
            filtered,
            x="Segment_Nama",
            y="AccountBalance",
            color="Segment_Nama",
            title="Sebaran Saldo per Segmen",
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Segmen",
            yaxis_title="Saldo Rekening",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # CHART 3 — USIA
    # ========================================================

    col3, col4 = st.columns(2)

    with col3:

        fig = px.histogram(
            filtered,
            x="CustomerAge",
            color="Segment_Nama",
            nbins=25,
            title="Distribusi Usia per Segmen",
            barmode="overlay",
        )

        fig.update_traces(
            opacity=0.65
        )

        fig.update_layout(
            xaxis_title="Usia",
            yaxis_title="Jumlah Nasabah",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # CHART 4 — CHANNEL
    # ========================================================

    with col4:

        channel_seg = (
            filtered
            .groupby(
                ["Segment_Nama", "Channel"]
            )
            .size()
            .reset_index(
                name="Jumlah"
            )
        )

        fig = px.bar(
            channel_seg,
            x="Segment_Nama",
            y="Jumlah",
            color="Channel",
            title="Channel Transaksi per Segmen",
            barmode="stack",
        )

        fig.update_layout(
            xaxis_title="Segmen",
            yaxis_title="Jumlah Transaksi",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # DATA TABLE
    # ========================================================

    st.subheader("📋 Data Setelah Filter")

    st.caption(
        f"Menampilkan maksimal 200 dari {len(filtered):,} baris."
    )

    st.dataframe(
        filtered.head(200),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.download_button(
        "⬇️ Download Data Hasil Filter",
        data=filtered.to_csv(
            index=False
        ),
        file_name="data_segmentasi_filtered.csv",
        mime="text/csv",
    )


# ============================================================
# ============================================================
# HALAMAN 2 — PREDIKSI INDIVIDUAL
# ============================================================
# ============================================================

elif page == "🔍 Prediksi Individual":

    st.title("🔍 Prediksi Kelompok Nasabah")

    st.markdown(
        """
        Masukkan karakteristik nasabah dan transaksi.
        Model kemudian akan memprediksi kelompok nasabah berdasarkan
        pola yang telah dipelajari.
        """
    )

    if not artifacts_loaded:

        st.error(
            "⚠️ Artifact model tidak ditemukan."
        )

        st.code(
            artifacts_error,
            language="text",
        )

        st.stop()

    # --------------------------------------------------------
    # Penjelasan
    # --------------------------------------------------------

    with st.expander(
        "ℹ️ Apa itu aplikasi ini?"
    ):

        st.markdown(
            """
            Aplikasi ini membagi nasabah bank ke dalam **4 kelompok**
            berdasarkan kemiripan karakteristik dan pola transaksi.

            Variabel yang digunakan antara lain:

            - Usia nasabah
            - Jumlah transaksi
            - Durasi transaksi
            - Percobaan login
            - Saldo rekening
            - Jenis transaksi
            - Channel transaksi
            - Pekerjaan nasabah
            """
        )

    st.divider()

    # ========================================================
    # PRESET
    # ========================================================

    st.subheader("⚡ Coba Contoh Cepat")

    st.caption(
        "Klik salah satu contoh untuk mengisi form secara otomatis."
    )

    presets = {

        "🧓 Nasabah Senior": {
            "amount": 250.0,
            "age": 65,
            "duration": 120,
            "login": 1,
            "balance": 8000.0,
            "ttype": "Credit",
            "channel": "Branch",
            "occ": "Retired",
        },

        "🎓 Mahasiswa": {
            "amount": 180.0,
            "age": 21,
            "duration": 90,
            "login": 1,
            "balance": 1200.0,
            "ttype": "Debit",
            "channel": "Branch",
            "occ": "Student",
        },

        "💼 Profesional": {
            "amount": 200.0,
            "age": 45,
            "duration": 110,
            "login": 1,
            "balance": 7000.0,
            "ttype": "Debit",
            "channel": "ATM",
            "occ": "Doctor",
        },

        "⚠️ Login Berulang": {
            "amount": 210.0,
            "age": 40,
            "duration": 130,
            "login": 4,
            "balance": 5500.0,
            "ttype": "Debit",
            "channel": "Online",
            "occ": "Doctor",
        },
    }

    if "form_values" not in st.session_state:

        st.session_state.form_values = {
            "amount": 250.0,
            "age": 35,
            "duration": 120,
            "login": 1,
            "balance": 5000.0,
            "ttype": "Debit",
            "channel": "ATM",
            "occ": "Doctor",
        }

    preset_cols = st.columns(
        len(presets)
    )

    for col, (label, values) in zip(
        preset_cols,
        presets.items(),
    ):

        if col.button(
            label,
            use_container_width=True,
        ):

            st.session_state.form_values = values

            st.rerun()

    st.divider()

    # ========================================================
    # INPUT FORM
    # ========================================================

    fv = st.session_state.form_values

    st.subheader("📝 Data Nasabah")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Usia nasabah",
            min_value=18,
            max_value=80,
            value=int(fv["age"]),
        )

        amount = st.slider(
            "Jumlah transaksi (Rp ribuan)",
            min_value=0,
            max_value=1000,
            value=int(fv["amount"]),
            step=10,
        )

        balance = st.slider(
            "Saldo rekening (Rp ribuan)",
            min_value=0,
            max_value=15000,
            value=int(fv["balance"]),
            step=100,
        )

        duration = st.slider(
            "Lama transaksi (detik)",
            min_value=10,
            max_value=300,
            value=int(fv["duration"]),
            step=5,
        )

    with col2:

        login = st.slider(
            "Percobaan login sebelum berhasil",
            min_value=1,
            max_value=5,
            value=int(fv["login"]),
        )

        transaction_classes = (
            label_encoders[
                "TransactionType"
            ]
            .classes_
            .tolist()
        )

        channel_classes = (
            label_encoders[
                "Channel"
            ]
            .classes_
            .tolist()
        )

        occupation_classes = (
            label_encoders[
                "CustomerOccupation"
            ]
            .classes_
            .tolist()
        )

        # Pastikan default masih tersedia
        default_ttype = (
            fv["ttype"]
            if fv["ttype"] in transaction_classes
            else transaction_classes[0]
        )

        default_channel = (
            fv["channel"]
            if fv["channel"] in channel_classes
            else channel_classes[0]
        )

        default_occ = (
            fv["occ"]
            if fv["occ"] in occupation_classes
            else occupation_classes[0]
        )

        ttype = st.radio(
            "Jenis transaksi",
            transaction_classes,
            index=transaction_classes.index(
                default_ttype
            ),
            horizontal=True,
        )

        channel = st.radio(
            "Channel transaksi",
            channel_classes,
            index=channel_classes.index(
                default_channel
            ),
            horizontal=True,
        )

        occ = st.selectbox(
            "Pekerjaan nasabah",
            occupation_classes,
            index=occupation_classes.index(
                default_occ
            ),
        )

    st.write("")

    predict_clicked = st.button(
        "🔍 Prediksi Kelompok Nasabah",
        type="primary",
        use_container_width=True,
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_clicked:

        try:

            input_dict = {

                "TransactionAmount":
                    float(amount),

                "CustomerAge":
                    int(age),

                "TransactionDuration":
                    int(duration),

                "LoginAttempts":
                    int(login),

                "AccountBalance":
                    float(balance),

                "TransactionType_encoded":
                    label_encoders[
                        "TransactionType"
                    ].transform([ttype])[0],

                "Channel_encoded":
                    label_encoders[
                        "Channel"
                    ].transform([channel])[0],

                "CustomerOccupation_encoded":
                    label_encoders[
                        "CustomerOccupation"
                    ].transform([occ])[0],
            }

            input_df = pd.DataFrame(
                [input_dict]
            )

            missing_input_features = [
                col
                for col in feature_cols
                if col not in input_df.columns
            ]

            if missing_input_features:

                st.error(
                    "Fitur input tidak sesuai dengan model:"
                )

                st.write(
                    missing_input_features
                )

                st.stop()

            X_input = input_df[
                feature_cols
            ]

            X_scaled = clf_scaler.transform(
                X_input
            )

            prediction = classifier.predict(
                X_scaled
            )

            pred_cluster = int(
                prediction[0]
            )

            info = get_segment_info(
                pred_cluster
            )

            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            confidence = None
            probabilities = None

            if hasattr(
                classifier,
                "predict_proba",
            ):

                probabilities = (
                    classifier
                    .predict_proba(X_scaled)[0]
                )

                confidence = (
                    float(
                        np.max(
                            probabilities
                        )
                    )
                    * 100
                )

            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🎯 Hasil Prediksi"
            )

            st.markdown(
                f"""
                <div class="prediction-card"
                     style="border-left: 6px solid {info['warna']};">

                    <div class="prediction-emoji">
                        {info['emoji']}
                    </div>

                    <div class="prediction-title"
                         style="color:{info['warna']};">
                        {info['nama']}
                    </div>

                    <div class="prediction-description">
                        {info['ringkas']}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            if confidence is not None:

                st.progress(
                    min(
                        int(confidence),
                        100,
                    ),
                    text=(
                        f"Tingkat keyakinan model: "
                        f"{confidence:.1f}%"
                    ),
                )

            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            st.subheader(
                "💡 Interpretasi"
            )

            st.write(
                info["detail"]
            )

            st.info(
                f"**Saran:** {info['saran']}"
            )

            # ------------------------------------------------
            # Detail teknis
            # ------------------------------------------------

            with st.expander(
                "🔧 Lihat Detail Teknis"
            ):

                st.markdown(
                    "**Data yang dimasukkan ke model:**"
                )

                st.dataframe(
                    X_input,
                    use_container_width=True,
                    hide_index=True,
                )

                if probabilities is not None:

                    classes = (
                        classifier.classes_
                        if hasattr(
                            classifier,
                            "classes_",
                        )
                        else range(
                            len(probabilities)
                        )
                    )

                    probability_rows = []

                    for class_id, probability in zip(
                        classes,
                        probabilities,
                    ):

                        class_id = int(
                            class_id
                        )

                        probability_rows.append(
                            {
                                "Kelompok":
                                    segment_label(
                                        class_id
                                    ),
                                "Kemungkinan (%)":
                                    round(
                                        float(
                                            probability
                                        )
                                        * 100,
                                        1,
                                    ),
                            }
                        )

                    probability_df = (
                        pd.DataFrame(
                            probability_rows
                        )
                        .sort_values(
                            "Kemungkinan (%)",
                            ascending=False,
                        )
                    )

                    st.markdown(
                        "**Probabilitas setiap kelompok:**"
                    )

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(
                    f"Model: "
                    f"{metadata.get('best_model_name', '-')}"
                    f" · Akurasi data uji: "
                    f"{metadata.get('best_model_accuracy', 0) * 100:.2f}%"
                )

        except Exception as e:

            st.error(
                "⚠️ Prediksi gagal dilakukan."
            )

            st.exception(e)


# ============================================================
# ============================================================
# HALAMAN 3 — PREDIKSI BATCH
# ============================================================
# ============================================================

elif page == "📁 Prediksi Batch (CSV)":

    st.title("📁 Prediksi Batch")

    st.markdown(
        """
        Unggah file CSV berisi beberapa nasabah untuk
        memprediksi segmen secara sekaligus.
        """
    )

    if not artifacts_loaded:

        st.error(
            "⚠️ Artifact model tidak ditemukan."
        )

        st.code(
            artifacts_error,
            language="text",
        )

        st.stop()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    st.subheader(
        "1️⃣ Format File"
    )

    st.info(
        "CSV harus memiliki kolom berikut:\n\n"
        + ", ".join(REQUIRED_BATCH_COLUMNS)
    )

    # --------------------------------------------------------
    # Template
    # --------------------------------------------------------

    template_df = pd.DataFrame(
        columns=REQUIRED_BATCH_COLUMNS
    )

    st.download_button(
        "⬇️ Download Template CSV",
        data=template_df.to_csv(
            index=False
        ),
        file_name="template_prediksi_batch.csv",
        mime="text/csv",
    )

    st.divider()

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    st.subheader(
        "2️⃣ Upload Data"
    )

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        help="Upload CSV sesuai format template.",
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"File berhasil dibaca: "
                f"{len(batch_df):,} baris."
            )

            # ------------------------------------------------
            # Validate columns
            # ------------------------------------------------

            missing_cols = [
                col
                for col in REQUIRED_BATCH_COLUMNS
                if col not in batch_df.columns
            ]

            if missing_cols:

                st.error(
                    "❌ Kolom berikut tidak ditemukan:"
                )

                st.write(
                    missing_cols
                )

                st.info(
                    "Gunakan template CSV yang tersedia di atas."
                )

                st.stop()

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            with st.expander(
                "👀 Preview Data",
                expanded=True,
            ):

                st.dataframe(
                    batch_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

            # ------------------------------------------------
            # Check numeric
            # ------------------------------------------------

            numeric_error = []

            for col in NUMERIC_COLUMNS:

                converted = pd.to_numeric(
                    batch_df[col],
                    errors="coerce",
                )

                if converted.isna().any():

                    numeric_error.append(
                        col
                    )

            if numeric_error:

                st.error(
                    "❌ Terdapat nilai non-numerik "
                    "pada kolom:"
                )

                st.write(
                    numeric_error
                )

                st.stop()

            # ------------------------------------------------
            # Encode
            # ------------------------------------------------

            df_encoded = encode_row(
                batch_df[
                    REQUIRED_BATCH_COLUMNS
                ],
                label_encoders,
            )

            encoded_columns = [
                col
                for col in feature_cols
                if col.endswith(
                    "_encoded"
                )
            ]

            # ------------------------------------------------
            # Unknown category
            # ------------------------------------------------

            unknown_mask = (
                df_encoded[
                    encoded_columns
                ]
                .isna()
                .any(axis=1)
            )

            valid_rows = (
                df_encoded[
                    ~unknown_mask
                ]
                .copy()
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            if valid_rows.empty:

                st.error(
                    "Tidak ada baris yang dapat diprediksi."
                )

                st.warning(
                    "Kemungkinan semua data memiliki "
                    "kategori yang tidak dikenali model."
                )

                st.stop()

            X_batch = valid_rows[
                feature_cols
            ]

            X_scaled = (
                clf_scaler.transform(
                    X_batch
                )
            )

            preds = (
                classifier
                .predict(X_scaled)
                .astype(int)
            )

            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            if hasattr(
                classifier,
                "predict_proba",
            ):

                probabilities = (
                    classifier
                    .predict_proba(
                        X_scaled
                    )
                    .max(axis=1)
                    * 100
                )

            else:

                probabilities = None

            # ------------------------------------------------
            # Result dataframe
            # ------------------------------------------------

            result_df = batch_df.copy()

            result_df["segmen"] = (
                "Tidak dikenali"
            )

            result_df.loc[
                valid_rows.index,
                "segmen",
            ] = [
                segment_label(
                    prediction
                )
                for prediction in preds
            ]

            if probabilities is not None:

                result_df["keyakinan_%"] = np.nan

                result_df.loc[
                    valid_rows.index,
                    "keyakinan_%",
                ] = np.round(
                    probabilities,
                    1,
                )

            # =================================================
            # RESULT SUMMARY
            # =================================================

            st.divider()

            st.subheader(
                "3️⃣ Hasil Prediksi"
            )

            total_rows = len(
                batch_df
            )

            valid_count = len(
                valid_rows
            )

            invalid_count = (
                total_rows
                - valid_count
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Data",
                f"{total_rows:,}",
            )

            c2.metric(
                "Berhasil Diprediksi",
                f"{valid_count:,}",
            )

            c3.metric(
                "Tidak Dikenali",
                f"{invalid_count:,}",
            )

            if unknown_mask.any():

                st.warning(
                    f"⚠️ {unknown_mask.sum():,} baris "
                    "memiliki kategori yang tidak dikenali "
                    "oleh model dan tidak diprediksi."
                )

            else:

                st.success(
                    "✅ Semua data berhasil diprediksi."
                )

            # ------------------------------------------------
            # Segment summary
            # ------------------------------------------------

            st.subheader(
                "📊 Distribusi Hasil Prediksi"
            )

            seg_summary = (
                result_df.loc[
                    valid_rows.index,
                    "segmen",
                ]
                .value_counts()
                .reset_index()
            )

            seg_summary.columns = [
                "Segmen",
                "Jumlah",
            ]

            col1, col2 = st.columns(2)

            with col1:

                st.dataframe(
                    seg_summary,
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:

                fig = px.pie(
                    seg_summary,
                    names="Segmen",
                    values="Jumlah",
                    title="Distribusi Segmen",
                    hole=0.35,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            # ------------------------------------------------
            # Full result
            # ------------------------------------------------

            st.subheader(
                "📋 Detail Hasil Prediksi"
            )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            st.download_button(
                "⬇️ Download Hasil Prediksi",
                data=result_df.to_csv(
                    index=False
                ),
                file_name="hasil_prediksi_batch.csv",
                mime="text/csv",
            )

        except pd.errors.EmptyDataError:

            st.error(
                "❌ File CSV kosong."
            )

        except pd.errors.ParserError:

            st.error(
                "❌ Format CSV tidak dapat dibaca. "
                "Periksa kembali delimiter dan struktur file."
            )

        except Exception as e:

            st.error(
                "⚠️ Terjadi kesalahan saat memproses file."
            )

            st.exception(e)


# ============================================================
# ============================================================
# HALAMAN 4 — MODEL INSIGHT
# ============================================================
# ============================================================

elif page == "📈 Model Insight":

    st.title("📈 Model Insight")

    st.markdown(
        """
        Halaman ini menampilkan model yang digunakan,
        performa model, feature importance, serta karakteristik
        setiap segmen nasabah.
        """
    )

    st.divider()

    # ========================================================
    # MODEL
    # ========================================================

    if artifacts_loaded:

        st.subheader(
            "🤖 Model yang Digunakan"
        )

        best_model = metadata.get(
            "best_model_name",
            "-",
        )

        accuracy = metadata.get(
            "best_model_accuracy",
            0,
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Model Terbaik",
            best_model,
        )

        c2.metric(
            "Akurasi Data Uji",
            f"{accuracy * 100:.2f}%",
        )

        st.caption(
            "Model terbaik dipilih berdasarkan performa "
            "akurasi pada data pengujian."
        )

        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        st.subheader(
            "📊 Perbandingan Model"
        )

        eval_df = pd.DataFrame(
            {
                "Model": [
                    "Random Forest",
                    "Decision Tree",
                ],
                "Akurasi Sebelum Tuning": [
                    "99.17%",
                    "99.58%",
                ],
                "Akurasi Setelah Tuning": [
                    "99.375%",
                    "99.58%",
                ],
            }
        )

        st.dataframe(
            eval_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Model dibandingkan menggunakan Random Forest "
            "dan Decision Tree dengan tuning menggunakan "
            "GridSearchCV 5-fold."
        )

        st.divider()

        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        st.subheader(
            "🎯 Feature Importance"
        )

        if hasattr(
            classifier,
            "feature_importances_",
        ):

            importance_values = (
                classifier
                .feature_importances_
            )

            if len(
                importance_values
            ) == len(feature_cols):

                fi_df = pd.DataFrame(
                    {
                        "Feature":
                            feature_cols,

                        "Importance":
                            importance_values,
                    }
                ).sort_values(
                    "Importance",
                    ascending=False,
                )

                fig = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=(
                        "Fitur yang Paling "
                        "Berpengaruh terhadap Prediksi"
                    ),
                )

                fig.update_layout(
                    yaxis={
                        "categoryorder":
                            "total ascending"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.dataframe(
                    fi_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.warning(
                    "Jumlah feature importance tidak sesuai "
                    "dengan jumlah feature pada feature_cols."
                )

        else:

            st.info(
                "Model terbaik tidak menyediakan "
                "feature importance."
            )

    else:

        st.error(
            "⚠️ Folder `artifacts/` tidak ditemukan."
        )

        st.code(
            artifacts_error,
            language="text",
        )

    # ========================================================
    # SEGMENT PROFILE
    # ========================================================

    st.divider()

    st.subheader(
        "👥 Profil Setiap Segmen"
    )

    for segment_id, info in SEGMENT_INFO.items():

        with st.expander(
            f"{info['emoji']} {info['nama']}"
        ):

            st.markdown(
                f"**Ringkasan:** {info['ringkas']}"
            )

            st.write(
                info["detail"]
            )

            st.info(
                f"💡 **Saran:** {info['saran']}"
            )

    # ========================================================
    # IMPORTANT NOTE
    # ========================================================

    st.divider()

    st.subheader(
        "ℹ️ Catatan Interpretasi Model"
    )

    st.info(
        """
        Label segmen pada dashboard berasal dari proses K-Means,
        kemudian digunakan sebagai target untuk melatih model
        klasifikasi.

        Karena target klasifikasi berasal dari hasil clustering
        dan fitur yang digunakan dalam klasifikasi berkaitan
        langsung dengan proses pembentukan cluster, akurasi model
        dapat menjadi sangat tinggi.

        Oleh karena itu, akurasi >99% tidak boleh langsung
        diinterpretasikan sebagai bukti bahwa model mampu
        memprediksi perilaku nasabah di dunia nyata dengan
        tingkat akurasi yang sama.
        """
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Dashboard Segmentasi Nasabah Bank
        · Customer Segmentation & Predictive Analytics
    </div>
    """,
    unsafe_allow_html=True,
)
