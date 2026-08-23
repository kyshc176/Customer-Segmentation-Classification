# ============================================================
# 1. IMPORT
# ============================================================
import json
import time
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
    page_title="Dashboard Segmentasi Nasabah",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 4. CSS MODERN UI/UX
# ============================================================
st.markdown(
    """
    <style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        border-radius: 12px;
        padding: 15px 20px;
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(128, 128, 128, 0.5);
    }

    /* Result card */
    .prediction-card {
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        background: rgba(255, 255, 255, 0.02);
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        transition: all 0.3s ease;
    }
    .prediction-card:hover {
        box-shadow: 0 15px 25px rgba(0,0,0,0.1);
    }
    .prediction-emoji {
        font-size: 50px;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .prediction-description {
        font-size: 16px;
        color: #777;
        line-height: 1.5;
    }

    /* Divider & Footer */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: rgba(128, 128, 128, 0.2);
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 14px;
        padding-top: 40px;
        font-weight: 500;
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
        "ringkas": "Nasabah usia lebih dewasa dengan kondisi keuangan yang stabil.",
        "detail": "Kelompok ini biasanya nasabah berusia lebih tua, sudah tidak lagi aktif bekerja, dengan saldo tabungan yang cukup besar. Mereka lebih suka datang langsung ke kantor cabang.",
        "saran": "Tawarkan produk deposito, investasi jangka panjang, atau layanan prioritas di cabang.",
    },
    1: {
        "nama": "Nasabah Muda / Pelajar",
        "emoji": "🎓",
        "warna": "#185FA5",
        "ringkas": "Nasabah usia muda, kemungkinan besar masih berstatus pelajar atau mahasiswa.",
        "detail": "Saldo tabungan kelompok ini relatif kecil, tetapi mereka cukup aktif bertransaksi dibandingkan jumlah saldo yang dimiliki.",
        "saran": "Tawarkan tabungan pelajar, promo cashback e-wallet, atau edukasi pengelolaan keuangan.",
    },
    2: {
        "nama": "Nasabah Profesional",
        "emoji": "💼",
        "warna": "#534AB7",
        "ringkas": "Kelompok terbesar — nasabah dengan pekerjaan tetap dan kondisi keuangan mapan.",
        "detail": "Kelompok ini mencakup nasabah usia menengah ke atas yang sudah bekerja tetap, dengan saldo tinggi namun transaksi cenderung rutin dan tidak besar-besar. Mereka lebih suka bertransaksi lewat ATM.",
        "saran": "Tawarkan kartu kredit, program cicilan, atau produk investasi menengah reksadana.",
    },
    3: {
        "nama": "Perlu Perhatian Keamanan",
        "emoji": "⚠️",
        "warna": "#A32D2D",
        "ringkas": "Nasabah dengan pola percobaan login yang jauh lebih sering dari biasanya.",
        "detail": "Kelompok kecil ini memiliki satu ciri khas: mereka mencoba login berkali-kali sebelum berhasil masuk ke akun (rata-rata 3–5 kali).",
        "saran": "Pantau aktivitas akun secara berkala. Kirimkan email edukasi terkait keamanan kata sandi atau biometrik login.",
    },
}

CATEGORICAL_COLUMNS = ["TransactionType", "Channel", "CustomerOccupation"]
REQUIRED_BATCH_COLUMNS = ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance", "TransactionType", "Channel", "CustomerOccupation"]
NUMERIC_COLUMNS = ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance"]

# ============================================================
# 6. CACHE & HELPER FUNCTIONS
# ============================================================
@st.cache_resource
def load_artifacts():
    required_files = [SCALER_PATH, ENCODER_PATH, MODEL_PATH, FEATURE_PATH, METADATA_PATH]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Artifact berikut tidak ditemukan:\n" + "\n".join(missing_files))
    
    return (
        joblib.load(SCALER_PATH),
        joblib.load(ENCODER_PATH),
        joblib.load(MODEL_PATH),
        list(joblib.load(FEATURE_PATH)),
        json.load(open(METADATA_PATH, "r", encoding="utf-8"))
    )

@st.cache_data
def load_raw_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def encode_row(df_raw, label_encoders):
    df = df_raw.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            known_values = set(le.classes_)
            df[f"{col}_encoded"] = df[col].apply(lambda x: le.transform([x])[0] if x in known_values else np.nan)
    return df

def get_segment_info(segment_id):
    return SEGMENT_INFO.get(int(segment_id), {"nama": f"Kelompok {segment_id}", "emoji": "❔", "warna": "#888888", "ringkas": "", "detail": "", "saran": ""})

def segment_label(segment_id):
    info = get_segment_info(segment_id)
    return f"{info['emoji']} {info['nama']}"

# ============================================================
# 7. INIT DATA & ARTIFACTS
# ============================================================
artifacts_loaded, data_loaded = True, True
artifacts_error, data_error = None, None

try:
    clf_scaler, label_encoders, classifier, feature_cols, metadata = load_artifacts()
except Exception as e:
    artifacts_loaded, artifacts_error = False, str(e)

try:
    raw_df = load_raw_data()
except Exception as e:
    data_loaded, data_error = False, str(e)

# ============================================================
# 8. SIDEBAR NAVIGASI
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60) # Contoh ikon bank
    st.title("Bank Analytics")
    
    page = st.radio(
        "Pilih Menu",
        ["📊 Overview & EDA", "🔍 Prediksi Individual", "📁 Prediksi Batch", "📈 Model Insight"],
        label_visibility="collapsed"
    )

    st.divider()
    
    if artifacts_loaded:
        st.markdown("**🧠 Status Model**")
        st.success("Online & Siap Digunakan")
        st.caption(f"**Akurasi:** {metadata.get('best_model_accuracy', 0) * 100:.2f}%")
        st.caption("**Algoritma:** Random Forest + K-Means")
    else:
        st.error("Model Offline")

# ============================================================
# HALAMAN 1 — OVERVIEW & EDA
# ============================================================
if page == "📊 Overview & EDA":
    st.title("📊 Overview Nasabah & EDA")
    st.markdown("Visualisasi interaktif untuk memahami distribusi perilaku nasabah bank Anda.")

    if not data_loaded or not artifacts_loaded:
        st.error("Data atau Model tidak tersedia. Periksa folder `artifacts` dan dataset.")
        st.stop()

    with st.spinner("Memproses segmentasi seluruh data..."):
        df_encoded = encode_row(raw_df, label_encoders)
        valid_mask = ~df_encoded[feature_cols].isna().any(axis=1)
        valid = df_encoded.loc[valid_mask].copy()
        
        X_scaled = clf_scaler.transform(valid[feature_cols])
        valid["Segment"] = classifier.predict(X_scaled).astype(int)
        valid["Segment_Nama"] = valid["Segment"].apply(segment_label)

    # Filter Data Miring (Expander lebih bersih)
    with st.expander("🔍 Filter Data Analisis", expanded=False):
        c1, c2, c3 = st.columns(3)
        seg_filter = c1.multiselect("Segmen", options=sorted(valid["Segment_Nama"].dropna().unique()))
        occ_filter = c2.multiselect("Pekerjaan", options=sorted(valid["CustomerOccupation"].dropna().unique()))
        min_a, max_a = int(valid["CustomerAge"].min()), int(valid["CustomerAge"].max())
        age_range = c3.slider("Rentang Usia", min_a, max_a, (min_a, max_a))

    filtered = valid.copy()
    if seg_filter: filtered = filtered[filtered["Segment_Nama"].isin(seg_filter)]
    if occ_filter: filtered = filtered[filtered["CustomerOccupation"].isin(occ_filter)]
    filtered = filtered[(filtered["CustomerAge"] >= age_range[0]) & (filtered["CustomerAge"] <= age_range[1])]

    # Metrik Kartu
    st.markdown("### 📈 Ringkasan Metrik (Data Difilter)")
    m1, m2, m3, m4 = st.columns(4)
    if filtered.empty:
        st.warning("Tidak ada data yang sesuai dengan filter.")
    else:
        m1.metric("Total Nasabah", f"{len(filtered):,}")
        m2.metric("Rata-rata Usia", f"{filtered['CustomerAge'].mean():.0f} th")
        m3.metric("Rata-rata Saldo", f"Rp {filtered['AccountBalance'].mean():,.0f}")
        m4.metric("Avg. Percobaan Login", f"{filtered['LoginAttempts'].mean():.1f}x")

    st.divider()

    # Menggunakan Tabs untuk Grafik agar tidak terlalu panjang ke bawah
    tab1, tab2, tab3 = st.tabs(["🧩 Distribusi Segmen", "💰 Keuangan & Transaksi", "📋 Data Lengkap"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            seg_count = filtered["Segment_Nama"].value_counts().reset_index()
            seg_count.columns = ["Segmen", "Jumlah"]
            fig1 = px.pie(seg_count, names="Segmen", values="Jumlah", title="Proporsi Segmen Nasabah", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig1.update_traces(textinfo="percent+label", textposition="inside")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.histogram(filtered, x="CustomerAge", color="Segment_Nama", nbins=20, title="Distribusi Usia Berdasarkan Segmen", barmode="overlay")
            fig2.update_traces(opacity=0.75)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.box(filtered, x="Segment_Nama", y="AccountBalance", color="Segment_Nama", title="Sebaran Saldo Rekening per Segmen")
            fig3.update_layout(showlegend=False, xaxis_title="Segmen", yaxis_title="Saldo Rekening")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            channel_seg = filtered.groupby(["Segment_Nama", "Channel"]).size().reset_index(name="Jumlah")
            fig4 = px.bar(channel_seg, x="Segment_Nama", y="Jumlah", color="Channel", title="Preferensi Channel Transaksi", barmode="group")
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export Data Filtered (CSV)", data=filtered.to_csv(index=False), file_name="data_segmentasi.csv", mime="text/csv")


# ============================================================
# HALAMAN 2 — PREDIKSI INDIVIDUAL
# ============================================================
elif page == "🔍 Prediksi Individual":
    st.title("🔍 Prediksi Nasabah Baru")
    st.markdown("Identifikasi persona nasabah dengan memasukkan data transaksi dan demografi di bawah ini.")
    
    if not artifacts_loaded: st.stop()

    # Preset Interaktif dengan tampilan pill/button modern
    st.markdown("#### ⚡ Coba Preset Cepat")
    preset_cols = st.columns(4)
    presets = {
        "🧓 Senior": {"amount": 250.0, "age": 65, "duration": 120, "login": 1, "balance": 8000.0, "ttype": "Credit", "channel": "Branch", "occ": "Retired"},
        "🎓 Mahasiswa": {"amount": 180.0, "age": 21, "duration": 90, "login": 1, "balance": 1200.0, "ttype": "Debit", "channel": "Branch", "occ": "Student"},
        "💼 Profesional": {"amount": 200.0, "age": 45, "duration": 110, "login": 1, "balance": 7000.0, "ttype": "Debit", "channel": "ATM", "occ": "Doctor"},
        "⚠️ Isu Keamanan": {"amount": 210.0, "age": 40, "duration": 130, "login": 4, "balance": 5500.0, "ttype": "Debit", "channel": "Online", "occ": "Doctor"},
    }

    if "form_values" not in st.session_state:
        st.session_state.form_values = presets["💼 Profesional"]

    for col, (label, values) in zip(preset_cols, presets.items()):
        if col.button(label, use_container_width=True):
            st.session_state.form_values = values
            st.rerun()

    st.divider()
    fv = st.session_state.form_values

    # Menggunakan Layout yang lebih bersih untuk form
    with st.container(border=True):
        st.markdown("### 📝 Masukkan Data Nasabah")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**👤 Data Demografi & Akun**")
            age = st.number_input("Usia Nasabah", min_value=18, max_value=100, value=int(fv["age"]))
            balance = st.number_input("Saldo Rekening (Rp Ribuan)", min_value=0, max_value=50000, value=int(fv["balance"]), step=500)
            occ = st.selectbox("Pekerjaan Nasabah", label_encoders["CustomerOccupation"].classes_, index=list(label_encoders["CustomerOccupation"].classes_).index(fv["occ"]))
            login = st.slider("Percobaan Login Gagal/Berhasil", 1, 5, int(fv["login"]), help="Jumlah rata-rata klik/percobaan saat login")

        with col2:
            st.markdown("**💳 Perilaku Transaksi**")
            amount = st.number_input("Rata-rata Transaksi (Rp Ribuan)", min_value=0, max_value=5000, value=int(fv["amount"]), step=50)
            duration = st.slider("Lama Transaksi (Detik)", 10, 300, int(fv["duration"]))
            ttype = st.radio("Jenis Transaksi Favorit", label_encoders["TransactionType"].classes_, index=list(label_encoders["TransactionType"].classes_).index(fv["ttype"]), horizontal=True)
            channel = st.radio("Channel Transaksi Favorit", label_encoders["Channel"].classes_, index=list(label_encoders["Channel"].classes_).index(fv["channel"]), horizontal=True)

        st.write("")
        predict_clicked = st.button("🚀 Analisis Segmentasi", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Menganalisis pola nasabah..."):
            time.sleep(0.5) # UX touch: memberikan efek model sedang "berpikir"
            try:
                input_df = pd.DataFrame([{
                    "TransactionAmount": float(amount), "CustomerAge": int(age), 
                    "TransactionDuration": int(duration), "LoginAttempts": int(login), "AccountBalance": float(balance),
                    "TransactionType_encoded": label_encoders["TransactionType"].transform([ttype])[0],
                    "Channel_encoded": label_encoders["Channel"].transform([channel])[0],
                    "CustomerOccupation_encoded": label_encoders["CustomerOccupation"].transform([occ])[0],
                }])

                X_scaled = clf_scaler.transform(input_df[feature_cols])
                pred_cluster = int(classifier.predict(X_scaled)[0])
                info = get_segment_info(pred_cluster)
                
                probabilities = classifier.predict_proba(X_scaled)[0] if hasattr(classifier, "predict_proba") else None

                # HASIL PREDIKSI (UX Card)
                st.markdown(
                    f"""
                    <div class="prediction-card" style="border-left: 8px solid {info['warna']};">
                        <div class="prediction-emoji">{info['emoji']}</div>
                        <div class="prediction-title" style="color:{info['warna']};">{info['nama']}</div>
                        <div class="prediction-description"><strong>Insight:</strong> {info['ringkas']}</div>
                    </div>
                    """, unsafe_allow_html=True
                )

                if probabilities is not None:
                    confidence = float(np.max(probabilities)) * 100
                    st.progress(int(confidence)/100, text=f"Keyakinan Model: {confidence:.1f}%")

                # REKOMENDASI BOX
                st.info(f"💡 **Tindakan yang Disarankan:** {info['saran']}")

            except Exception as e:
                st.error("Terjadi kesalahan saat memprediksi data.")
                st.exception(e)


# ============================================================
# HALAMAN 3 — PREDIKSI BATCH
# ============================================================
elif page == "📁 Prediksi Batch":
    st.title("📁 Prediksi Banyak Nasabah Sekaligus (Batch)")
    st.markdown("Gunakan fitur ini jika tim marketing perlu menyortir ribuan data nasabah baru untuk penawaran campaign.")

    if not artifacts_loaded: st.stop()

    st.markdown("### Langkah 1: Siapkan File CSV")
    st.caption(f"Kolom wajib: {', '.join(REQUIRED_BATCH_COLUMNS)}")
    template_df = pd.DataFrame(columns=REQUIRED_BATCH_COLUMNS)
    st.download_button("⬇️ Download Template CSV", data=template_df.to_csv(index=False), file_name="template_batch.csv", mime="text/csv")

    st.markdown("### Langkah 2: Upload Data")
    uploaded_file = st.file_uploader("Seret dan jatuhkan file CSV ke sini", type=["csv"])

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Berhasil memuat {len(batch_df):,} baris data.")
            st.toast("File berhasil diupload! Memproses data...", icon="⏳")

            # Validate & Process
            df_encoded = encode_row(batch_df[REQUIRED_BATCH_COLUMNS], label_encoders)
            encoded_columns = [col for col in feature_cols if col.endswith("_encoded")]
            unknown_mask = df_encoded[encoded_columns].isna().any(axis=1)
            valid_rows = df_encoded[~unknown_mask].copy()

            if valid_rows.empty:
                st.error("Semua data gagal diproses (kategori tidak dikenali model).")
                st.stop()

            # Prediksi
            X_scaled = clf_scaler.transform(valid_rows[feature_cols])
            preds = classifier.predict(X_scaled).astype(int)
            
            result_df = batch_df.copy()
            result_df["Hasil_Segmen"] = "Tidak dikenali"
            result_df.loc[valid_rows.index, "Hasil_Segmen"] = [segment_label(p) for p in preds]

            if hasattr(classifier, "predict_proba"):
                probs = classifier.predict_proba(X_scaled).max(axis=1) * 100
                result_df["Keyakinan_Model(%)"] = np.nan
                result_df.loc[valid_rows.index, "Keyakinan_Model(%)"] = np.round(probs, 1)

            st.balloons()
            st.toast("Prediksi selesai!", icon="🎉")

            st.markdown("### Langkah 3: Hasil Analisis")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Baris", f"{len(batch_df):,}")
            c2.metric("Sukses Diprediksi", f"{len(valid_rows):,}")
            c3.metric("Gagal Diproses", f"{len(batch_df) - len(valid_rows):,}")

            st.dataframe(result_df, use_container_width=True, hide_index=True)

            st.download_button("⬇️ Download Hasil Analisis (CSV)", data=result_df.to_csv(index=False), file_name="hasil_prediksi.csv", mime="text/csv", type="primary")

        except Exception as e:
            st.error("Terjadi kesalahan membaca file. Pastikan format sesuai template.")
            st.exception(e)


# ============================================================
# HALAMAN 4 — MODEL INSIGHT
# ============================================================
elif page == "📈 Model Insight":
    st.title("🧠 Cara Kerja Model (XAI)")
    st.markdown("Memahami alasan mengapa model mengelompokkan nasabah ke segmen tertentu (Explainable AI).")

    if not artifacts_loaded: st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏆 Performa Model")
        st.info(f"**Model Utama:** {metadata.get('best_model_name', 'Random Forest')}")
        st.metric("Akurasi Data Pengujian", f"{metadata.get('best_model_accuracy', 0) * 100:.2f}%")
        st.caption("Target label dihasilkan dari K-Means Clustering (k=4).")

    with col2:
        st.markdown("### 🎯 Atribut Paling Berpengaruh")
        if hasattr(classifier, "feature_importances_"):
            fi_df = pd.DataFrame({
                "Fitur": feature_cols, "Skor Penting": classifier.feature_importances_
            }).sort_values("Skor Penting", ascending=True)
            
            fig = px.bar(fi_df, x="Skor Penting", y="Fitur", orientation="h", height=250)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Model tidak mendukung feature importance secara langsung.")

    st.divider()

    st.markdown("### 👥 Ensiklopedia Segmen Nasabah")
    for segment_id, info in SEGMENT_INFO.items():
        with st.expander(f"{info['emoji']} {info['nama']}", expanded=False):
            st.write(info['detail'])
            st.success(f"**Strategi Bisnis:** {info['saran']}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""<div class="footer">Dashboard Segmentasi Nasabah Bank · © 2026 Customer Intelligence Team</div>""", unsafe_allow_html=True)
