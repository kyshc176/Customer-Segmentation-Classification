"""
app.py — Dashboard Prediksi Kelompok Nasabah Bank

Dashboard multi-halaman: Overview & EDA, Prediksi Individual, Prediksi Batch (CSV),
dan Model Insight. Dibangun di atas model klasifikasi (hasil clustering K-Means yang
"diajarkan" ke Random Forest/Decision Tree) dari proyek Customer Segmentation.

Jalankan lokal:
    streamlit run app.py

Pastikan folder artifacts/ (hasil `python train_export.py`) dan
bank_transactions_data_2.csv ada di folder yang sama.
"""

import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Segmentasi Nasabah Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Load artifacts & data
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    clf_scaler = joblib.load("artifacts/clf_scaler.pkl")
    label_encoders = joblib.load("artifacts/label_encoders.pkl")
    classifier = joblib.load("artifacts/classifier_model.pkl")
    feature_cols = joblib.load("artifacts/feature_cols.pkl")
    with open("artifacts/metadata.json") as f:
        metadata = json.load(f)
    return clf_scaler, label_encoders, classifier, feature_cols, metadata


@st.cache_data
def load_raw_data():
    return pd.read_csv("bank_transactions_data_2.csv")


def encode_row(df_raw, label_encoders):
    """Encode kolom kategorikal jadi *_encoded sesuai label_encoders yang sudah di-fit."""
    df = df_raw.copy()
    for col in ["TransactionType", "Channel", "CustomerOccupation"]:
        le = label_encoders[col]
        known = set(le.classes_)
        df[f"{col}_encoded"] = df[col].apply(lambda v: le.transform([v])[0] if v in known else np.nan)
    return df


artifacts_loaded = True
try:
    clf_scaler, label_encoders, classifier, feature_cols, metadata = load_artifacts()
except FileNotFoundError:
    artifacts_loaded = False

data_loaded = True
try:
    raw_df = load_raw_data()
except FileNotFoundError:
    data_loaded = False

SEGMENT_INFO = {
    0: {"nama": "Nasabah Senior Mapan", "emoji": "🧓", "warna": "#0F6E56",
        "ringkas": "Nasabah usia lebih dewasa dengan kondisi keuangan yang stabil.",
        "detail": ("Kelompok ini biasanya nasabah berusia lebih tua, sudah tidak lagi aktif bekerja, "
                   "dengan saldo tabungan yang cukup besar. Mereka lebih suka datang langsung ke kantor cabang."),
        "saran": "Cocok ditawari produk deposito, investasi jangka panjang, atau layanan prioritas di cabang."},
    1: {"nama": "Nasabah Muda / Pelajar", "emoji": "🎓", "warna": "#185FA5",
        "ringkas": "Nasabah usia muda, kemungkinan besar masih berstatus pelajar/mahasiswa.",
        "detail": ("Saldo tabungan kelompok ini relatif kecil, tapi mereka cukup aktif bertransaksi "
                   "dibanding jumlah saldo yang dimiliki."),
        "saran": "Cocok ditawari tabungan pelajar, promo cashback, atau edukasi pengelolaan keuangan."},
    2: {"nama": "Nasabah Profesional", "emoji": "💼", "warna": "#534AB7",
        "ringkas": "Kelompok terbesar — nasabah dengan pekerjaan tetap dan kondisi keuangan mapan.",
        "detail": ("Kelompok ini mencakup nasabah usia menengah ke atas yang sudah bekerja tetap, "
                   "dengan saldo tinggi namun transaksi cenderung rutin dan tidak besar-besar. "
                   "Lebih suka bertransaksi lewat ATM."),
        "saran": "Cocok ditawari kartu kredit, cicilan, atau produk investasi menengah."},
    3: {"nama": "Perlu Perhatian Keamanan", "emoji": "⚠️", "warna": "#A32D2D",
        "ringkas": "Nasabah dengan pola percobaan login yang jauh lebih sering dari biasanya.",
        "detail": ("Kelompok kecil ini punya satu ciri khas: mereka mencoba login berkali-kali "
                   "sebelum berhasil masuk ke akun (rata-rata 3-5 kali, dibanding kelompok lain yang "
                   "biasanya langsung berhasil di percobaan pertama)."),
        "saran": "Sebaiknya dipantau lebih lanjut — bisa jadi nasabah lupa kata sandi, atau indikasi percobaan akses tidak sah."},
}

# ----------------------------------------------------------------------
# Sidebar navigasi
# ----------------------------------------------------------------------
st.sidebar.title("🏦 Dashboard Segmentasi Nasabah")
page = st.sidebar.radio(
    "Navigasi",
    ["📊 Overview & EDA", "🔍 Prediksi Individual", "📁 Prediksi Batch (CSV)", "📈 Model Insight"]
)
st.sidebar.divider()
if artifacts_loaded:
    st.sidebar.caption(f"Model: {metadata.get('best_model_name', '-')} · "
                        f"Akurasi: {metadata.get('best_model_accuracy', 0) * 100:.2f}%")
st.sidebar.caption("Data: bank_transactions_data_2.csv · Segmen dari K-Means (k=4)")

# =============================================================
# HALAMAN 1: OVERVIEW & EDA
# =============================================================
if page == "📊 Overview & EDA":
    st.title("📊 Overview & Exploratory Data Analysis")

    if not data_loaded:
        st.error("⚠️ File bank_transactions_data_2.csv tidak ditemukan di folder yang sama.")
    elif not artifacts_loaded:
        st.error("⚠️ Folder artifacts/ tidak ditemukan — jalankan `python train_export.py` terlebih dahulu.")
    else:
        with st.spinner("Menghitung segmen untuk seluruh nasabah..."):
            df_encoded = encode_row(raw_df, label_encoders)
            valid = df_encoded.dropna(subset=[c for c in feature_cols if c.endswith("_encoded")])
            X_all = valid[feature_cols]
            X_scaled = clf_scaler.transform(X_all)
            valid = valid.copy()
            valid["Segment"] = classifier.predict(X_scaled).astype(int)
            valid["Segment_Nama"] = valid["Segment"].map(lambda s: f"{SEGMENT_INFO[s]['emoji']} {SEGMENT_INFO[s]['nama']}")

        # --- Filter ---
        with st.expander("🔧 Filter Data", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                seg_filter = st.multiselect("Segmen", sorted(valid["Segment_Nama"].unique()))
            with c2:
                occ_filter = st.multiselect("Pekerjaan", sorted(valid["CustomerOccupation"].unique()))
            with c3:
                age_range = st.slider("Rentang Usia", int(valid["CustomerAge"].min()), int(valid["CustomerAge"].max()),
                                       (int(valid["CustomerAge"].min()), int(valid["CustomerAge"].max())))

        filtered = valid.copy()
        if seg_filter:
            filtered = filtered[filtered["Segment_Nama"].isin(seg_filter)]
        if occ_filter:
            filtered = filtered[filtered["CustomerOccupation"].isin(occ_filter)]
        filtered = filtered[(filtered["CustomerAge"] >= age_range[0]) & (filtered["CustomerAge"] <= age_range[1])]

        # --- Metrik ---
        st.subheader("Ringkasan")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Nasabah", f"{len(filtered):,}")
        m2.metric("Rata-rata Usia", f"{filtered['CustomerAge'].mean():.0f} th")
        m3.metric("Rata-rata Saldo", f"{filtered['AccountBalance'].mean():,.0f}")
        m4.metric("Rata-rata Percobaan Login", f"{filtered['LoginAttempts'].mean():.1f}")

        st.divider()

        # --- Charts ---
        col1, col2 = st.columns(2)
        with col1:
            seg_count = filtered["Segment_Nama"].value_counts().reset_index()
            seg_count.columns = ["Segmen", "Jumlah"]
            fig = px.pie(seg_count, names="Segmen", values="Jumlah", title="Distribusi Segmen Nasabah",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(filtered, x="Segment_Nama", y="AccountBalance", title="Sebaran Saldo per Segmen",
                         color="Segment_Nama", color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(filtered, x="CustomerAge", color="Segment_Nama", nbins=25,
                                title="Distribusi Usia per Segmen", barmode="overlay",
                                color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(opacity=0.65)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            channel_seg = filtered.groupby(["Segment_Nama", "Channel"]).size().reset_index(name="Jumlah")
            fig = px.bar(channel_seg, x="Segment_Nama", y="Jumlah", color="Channel", title="Channel Transaksi per Segmen",
                         barmode="stack", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data (setelah filter)")
        st.dataframe(filtered.head(200), use_container_width=True)

# =============================================================
# HALAMAN 2: PREDIKSI INDIVIDUAL
# =============================================================
elif page == "🔍 Prediksi Individual":
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>🏦 Prediksi Kelompok Nasabah</h1>"
        "<p style='text-align:center; color:gray; font-size:16px; margin-top:4px;'>"
        "Isi data transaksi di bawah, aplikasi akan menebak kelompok nasabah ini secara otomatis."
        "</p>", unsafe_allow_html=True,
    )

    if not artifacts_loaded:
        st.error("⚠️ Folder artifacts/ tidak ditemukan — jalankan `python train_export.py` terlebih dahulu.")
        st.stop()

    with st.expander("ℹ️ Apa itu aplikasi ini?"):
        st.markdown(
            """
            Aplikasi ini membagi nasabah bank ke dalam **4 kelompok** berdasarkan kemiripan pola transaksi
            mereka (usia, jumlah transaksi, saldo, dan sebagainya). Silakan isi data di bawah, atau klik
            salah satu **contoh cepat** untuk langsung mencoba.
            """
        )
    st.divider()

    st.markdown("**Coba contoh cepat** (opsional):")
    presets = {
        "🧓 Nasabah senior": dict(amount=250, age=65, duration=120, login=1, balance=8000,
                                   ttype="Credit", channel="Branch", occ="Retired"),
        "🎓 Mahasiswa": dict(amount=180, age=21, duration=90, login=1, balance=1200,
                              ttype="Debit", channel="Branch", occ="Student"),
        "💼 Profesional": dict(amount=200, age=45, duration=110, login=1, balance=7000,
                                occ="Doctor", channel="ATM", ttype="Debit"),
        "⚠️ Login berulang": dict(amount=210, age=40, duration=130, login=4, balance=5500,
                                   ttype="Debit", channel="Online", occ="Doctor"),
    }
    preset_cols = st.columns(len(presets))
    if "form_values" not in st.session_state:
        st.session_state.form_values = dict(
            amount=250.0, age=35, duration=120, login=1, balance=5000.0,
            ttype="Debit", channel="ATM", occ="Doctor",
        )
    for col, (label, values) in zip(preset_cols, presets.items()):
        if col.button(label, use_container_width=True):
            st.session_state.form_values = values

    st.divider()
    fv = st.session_state.form_values
    st.markdown("**Data nasabah**")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Usia nasabah", min_value=18, max_value=80, value=fv["age"])
        amount = st.slider("Jumlah uang di transaksi ini (Rp ribuan)", min_value=0, max_value=1000,
                            value=int(fv["amount"]), step=10)
        balance = st.slider("Saldo rekening saat ini (Rp ribuan)", min_value=0, max_value=15000,
                             value=int(fv["balance"]), step=100)
        duration = st.slider("Lama waktu transaksi (detik)", min_value=10, max_value=300, value=fv["duration"])
    with col2:
        login = st.slider("Berapa kali mencoba login sebelum berhasil?", min_value=1, max_value=5, value=fv["login"])
        ttype = st.radio("Jenis transaksi", label_encoders["TransactionType"].classes_.tolist(),
                          index=list(label_encoders["TransactionType"].classes_).index(fv["ttype"]), horizontal=True)
        channel = st.radio("Dilakukan lewat mana?", label_encoders["Channel"].classes_.tolist(),
                            index=list(label_encoders["Channel"].classes_).index(fv["channel"]), horizontal=True)
        occ = st.selectbox("Pekerjaan nasabah", label_encoders["CustomerOccupation"].classes_.tolist(),
                            index=list(label_encoders["CustomerOccupation"].classes_).index(fv["occ"]))

    st.write("")
    predict_clicked = st.button("🔍 Tebak kelompok nasabah ini", type="primary", use_container_width=True)

    if predict_clicked:
        input_dict = {
            "TransactionAmount": float(amount), "CustomerAge": int(age),
            "TransactionDuration": int(duration), "LoginAttempts": int(login),
            "AccountBalance": float(balance),
            "TransactionType_encoded": label_encoders["TransactionType"].transform([ttype])[0],
            "Channel_encoded": label_encoders["Channel"].transform([channel])[0],
            "CustomerOccupation_encoded": label_encoders["CustomerOccupation"].transform([occ])[0],
        }
        X_input = pd.DataFrame([input_dict])[feature_cols]
        X_scaled = clf_scaler.transform(X_input)
        pred_cluster = int(classifier.predict(X_scaled)[0])
        info = SEGMENT_INFO.get(pred_cluster, {"nama": f"Kelompok {pred_cluster}", "emoji": "❔", "warna": "#888",
                                                "ringkas": "", "detail": "", "saran": ""})
        confidence = float(max(classifier.predict_proba(X_scaled)[0])) * 100 if hasattr(classifier, "predict_proba") else None

        st.divider()
        st.markdown(
            f"""<div style="border:2px solid {info['warna']}22; border-left:6px solid {info['warna']};
            border-radius:12px; padding:20px 24px; background:{info['warna']}10;">
            <div style="font-size:38px; line-height:1;">{info['emoji']}</div>
            <div style="font-size:22px; font-weight:600; color:{info['warna']}; margin-top:6px;">{info['nama']}</div>
            <div style="font-size:15px; color:#555; margin-top:4px;">{info['ringkas']}</div></div>""",
            unsafe_allow_html=True,
        )
        if confidence is not None:
            st.write("")
            st.progress(min(int(confidence), 100), text=f"Tingkat keyakinan model: {confidence:.0f}%")
        st.write("")
        st.markdown(f"**Penjelasan:** {info['detail']}")
        st.info(f"💡 **Saran:** {info['saran']}")

        with st.expander("Lihat detail teknis (untuk yang penasaran)"):
            st.write("Data yang dimasukkan:")
            st.dataframe(X_input, hide_index=True)
            if hasattr(classifier, "predict_proba"):
                proba = classifier.predict_proba(X_scaled)[0]
                proba_df = pd.DataFrame({
                    "Kelompok": [f"{SEGMENT_INFO[i]['emoji']} {SEGMENT_INFO[i]['nama']}" for i in range(len(proba))],
                    "Kemungkinan (%)": (proba * 100).round(1),
                }).sort_values("Kemungkinan (%)", ascending=False)
                st.dataframe(proba_df, hide_index=True, use_container_width=True)
            st.caption(f"Model yang dipakai: {metadata['best_model_name']} · "
                       f"Akurasi pada data pengujian: {metadata['best_model_accuracy']*100:.2f}%")

# =============================================================
# HALAMAN 3: PREDIKSI BATCH (CSV)
# =============================================================
elif page == "📁 Prediksi Batch (CSV)":
    st.title("📁 Prediksi Batch")
    st.markdown("*Unggah file CSV berisi banyak nasabah untuk diprediksi segmennya sekaligus.*")

    if not artifacts_loaded:
        st.error("⚠️ Folder artifacts/ tidak ditemukan — jalankan `python train_export.py` terlebih dahulu.")
        st.stop()

    required_cols = ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts",
                      "AccountBalance", "TransactionType", "Channel", "CustomerOccupation"]
    st.info("Kolom yang dibutuhkan: " + ", ".join(required_cols))

    template_df = pd.DataFrame(columns=required_cols)
    st.download_button("⬇️ Download Template CSV", template_df.to_csv(index=False),
                        file_name="template_prediksi_batch.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in required_cols if c not in batch_df.columns]
            if missing_cols:
                st.error(f"⚠️ Kolom berikut tidak ditemukan di file: {missing_cols}")
            else:
                df_encoded = encode_row(batch_df[required_cols], label_encoders)
                unknown_mask = df_encoded[[c for c in feature_cols if c.endswith("_encoded")]].isna().any(axis=1)
                valid_rows = df_encoded[~unknown_mask]
                X_batch = valid_rows[feature_cols]
                X_scaled = clf_scaler.transform(X_batch)
                preds = classifier.predict(X_scaled).astype(int)
                proba = classifier.predict_proba(X_scaled).max(axis=1) * 100 if hasattr(classifier, "predict_proba") else None

                result_df = batch_df.copy()
                result_df["segmen"] = "Tidak dikenali (kategori di luar data latih)"
                result_df.loc[valid_rows.index, "segmen"] = [
                    f"{SEGMENT_INFO[p]['emoji']} {SEGMENT_INFO[p]['nama']}" for p in preds
                ]
                if proba is not None:
                    result_df.loc[valid_rows.index, "keyakinan_%"] = proba.round(1)

                st.success(f"✅ Berhasil memprediksi {len(valid_rows)} dari {len(batch_df)} nasabah.")
                if unknown_mask.any():
                    st.warning(f"⚠️ {unknown_mask.sum()} baris punya kategori (TransactionType/Channel/CustomerOccupation) "
                               "yang tidak dikenali model dan dilewati.")

                seg_summary = result_df.loc[valid_rows.index, "segmen"].value_counts().reset_index()
                seg_summary.columns = ["Segmen", "Jumlah"]
                st.dataframe(seg_summary, use_container_width=True, hide_index=True)

                st.dataframe(result_df, use_container_width=True)
                st.download_button("⬇️ Download Hasil Prediksi", result_df.to_csv(index=False),
                                    file_name="hasil_prediksi_batch.csv", mime="text/csv")
        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan saat memproses file: {e}")

# =============================================================
# HALAMAN 4: MODEL INSIGHT
# =============================================================
elif page == "📈 Model Insight":
    st.title("📈 Model Insight")
    st.markdown("*Performa model dan profil tiap segmen nasabah.*")
    st.divider()

    if artifacts_loaded:
        st.subheader("Model yang Dipakai")
        c1, c2 = st.columns(2)
        c1.metric("Model Terbaik", metadata.get("best_model_name", "-"))
        c2.metric("Akurasi (data uji)", f"{metadata.get('best_model_accuracy', 0) * 100:.2f}%")

        st.caption("Dua model dibandingkan (Random Forest & Decision Tree, dituning dengan GridSearchCV 5-fold) — "
                   "model dengan akurasi tertinggi dipilih otomatis untuk aplikasi ini.")

        eval_df = pd.DataFrame({
            "Model": ["Random Forest", "Decision Tree"],
            "Akurasi Sebelum Tuning": ["99.17%", "99.58%"],
            "Akurasi Setelah Tuning": ["99.375%", "99.58% (tidak berubah)"],
        })
        st.dataframe(eval_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Feature Importance")
        if hasattr(classifier, "feature_importances_"):
            fi_df = pd.DataFrame({
                "feature": feature_cols, "importance": classifier.feature_importances_
            }).sort_values("importance", ascending=False)
            fig = px.bar(fi_df, x="importance", y="feature", orientation="h", title="Fitur Paling Berpengaruh",
                         color_discrete_sequence=["#534AB7"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model terbaik saat ini tidak menyediakan feature importance (mungkin Decision Tree tanpa atribut ini, atau model lain).")
    else:
        st.error("⚠️ Folder artifacts/ tidak ditemukan.")

    st.divider()
    st.subheader("Profil Setiap Segmen")
    for seg_id, info in SEGMENT_INFO.items():
        with st.expander(f"{info['emoji']} {info['nama']}"):
            st.write(info["detail"])
            st.info(f"💡 **Saran:** {info['saran']}")

    st.caption(
        "Catatan: karena label segmen berasal dari K-Means pada fitur yang sama persis dengan fitur input "
        "klasifikasi, akurasi yang sangat tinggi (>99%) wajar secara matematis — bukan semata indikasi "
        "overfitting seperti pada klasifikasi dengan label independen/asli."
    )
