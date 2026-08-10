"""
app.py — Prediksi Kelompok Nasabah Bank
Versi UI yang dirancang untuk pengguna awam (non-teknis): bahasa sederhana,
slider dengan rentang wajar, contoh cepat untuk dicoba, dan hasil yang
dijelaskan dalam kalimat biasa (bukan istilah statistik).

Jalankan lokal:
    streamlit run app.py

Pastikan folder artifacts/ (hasil `python train_export.py`) ada di folder yang sama.
"""

import json

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Konfigurasi & load model
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Prediksi Kelompok Nasabah",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_artifacts():
    clf_scaler = joblib.load("artifacts/clf_scaler.pkl")
    label_encoders = joblib.load("artifacts/label_encoders.pkl")
    classifier = joblib.load("artifacts/classifier_model.pkl")
    feature_cols = joblib.load("artifacts/feature_cols.pkl")
    with open("artifacts/metadata.json") as f:
        metadata = json.load(f)
    return clf_scaler, label_encoders, classifier, feature_cols, metadata


try:
    clf_scaler, label_encoders, classifier, feature_cols, metadata = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model belum siap. Jalankan perintah `python train_export.py` terlebih dahulu "
        "di folder yang sama, baru buka aplikasi ini lagi."
    )
    st.stop()

# Deskripsi tiap kelompok dalam bahasa sederhana + rekomendasi tindakan
SEGMENT_INFO = {
    0: {
        "nama": "Nasabah Senior Mapan",
        "emoji": "🧓",
        "warna": "#0F6E56",
        "ringkas": "Nasabah usia lebih dewasa dengan kondisi keuangan yang stabil.",
        "detail": (
            "Kelompok ini biasanya nasabah berusia lebih tua, sudah tidak lagi aktif bekerja, "
            "dengan saldo tabungan yang cukup besar. Mereka lebih suka datang langsung ke kantor cabang."
        ),
        "saran": "Cocok ditawari produk deposito, investasi jangka panjang, atau layanan prioritas di cabang.",
    },
    1: {
        "nama": "Nasabah Muda / Pelajar",
        "emoji": "🎓",
        "warna": "#185FA5",
        "ringkas": "Nasabah usia muda, kemungkinan besar masih berstatus pelajar/mahasiswa.",
        "detail": (
            "Saldo tabungan kelompok ini relatif kecil, tapi mereka cukup aktif bertransaksi "
            "dibanding jumlah saldo yang dimiliki."
        ),
        "saran": "Cocok ditawari tabungan pelajar, promo cashback, atau edukasi pengelolaan keuangan.",
    },
    2: {
        "nama": "Nasabah Profesional",
        "emoji": "💼",
        "warna": "#534AB7",
        "ringkas": "Kelompok terbesar — nasabah dengan pekerjaan tetap dan kondisi keuangan mapan.",
        "detail": (
            "Kelompok ini mencakup nasabah usia menengah ke atas yang sudah bekerja tetap, "
            "dengan saldo tinggi namun transaksi cenderung rutin dan tidak besar-besar. "
            "Lebih suka bertransaksi lewat ATM."
        ),
        "saran": "Cocok ditawari kartu kredit, cicilan, atau produk investasi menengah.",
    },
    3: {
        "nama": "Perlu Perhatian Keamanan",
        "emoji": "⚠️",
        "warna": "#A32D2D",
        "ringkas": "Nasabah dengan pola percobaan login yang jauh lebih sering dari biasanya.",
        "detail": (
            "Kelompok kecil ini punya satu ciri khas: mereka mencoba login berkali-kali "
            "sebelum berhasil masuk ke akun (rata-rata 3-5 kali, dibanding kelompok lain yang "
            "biasanya langsung berhasil di percobaan pertama)."
        ),
        "saran": "Sebaiknya dipantau lebih lanjut — bisa jadi nasabah lupa kata sandi, atau indikasi percobaan akses tidak sah.",
    },
}

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>🏦 Prediksi Kelompok Nasabah</h1>"
    "<p style='text-align:center; color:var(--text-secondary, gray); font-size:16px; margin-top:4px;'>"
    "Isi data transaksi di bawah, aplikasi akan menebak kelompok nasabah ini secara otomatis."
    "</p>",
    unsafe_allow_html=True,
)

with st.expander("ℹ️ Apa itu aplikasi ini?"):
    st.markdown(
        """
Aplikasi ini membagi nasabah bank ke dalam **4 kelompok** berdasarkan kemiripan pola transaksi
mereka (usia, jumlah transaksi, saldo, dan sebagainya). Pengelompokan ini ditemukan otomatis
oleh komputer dari ribuan data transaksi sebelumnya, lalu "diajarkan" ke sebuah model prediksi
supaya bisa langsung menebak kelompok nasabah baru tanpa perlu mengumpulkan ribuan data lagi.

Silakan isi data di bawah, atau klik salah satu **contoh cepat** untuk langsung mencoba.
        """
    )

st.divider()

# ----------------------------------------------------------------------
# Contoh cepat (preset) — supaya orang awam tidak perlu mikir angka
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Form input — bahasa sederhana + slider
# ----------------------------------------------------------------------

fv = st.session_state.form_values

st.markdown("**Data nasabah**")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Usia nasabah", min_value=18, max_value=80, value=fv["age"],
        help="Usia nasabah dalam tahun.",
    )
    amount = st.slider(
        "Jumlah uang di transaksi ini (Rp ribuan)", min_value=0, max_value=1000,
        value=int(fv["amount"]), step=10,
        help="Nominal transaksi yang dilakukan nasabah, misalnya saat belanja atau transfer.",
    )
    balance = st.slider(
        "Saldo rekening saat ini (Rp ribuan)", min_value=0, max_value=15000,
        value=int(fv["balance"]), step=100,
        help="Total saldo yang dimiliki nasabah di rekeningnya.",
    )
    duration = st.slider(
        "Lama waktu transaksi (detik)", min_value=10, max_value=300,
        value=fv["duration"],
        help="Berapa lama proses transaksi berlangsung, dari mulai sampai selesai.",
    )

with col2:
    login = st.slider(
        "Berapa kali mencoba login sebelum berhasil?", min_value=1, max_value=5,
        value=fv["login"],
        help="Kebanyakan nasabah berhasil login di percobaan pertama. Angka yang lebih tinggi "
             "berarti ada beberapa kali percobaan gagal sebelum akhirnya berhasil masuk.",
    )
    ttype = st.radio(
        "Jenis transaksi", label_encoders["TransactionType"].classes_.tolist(),
        index=list(label_encoders["TransactionType"].classes_).index(fv["ttype"]),
        horizontal=True,
    )
    channel = st.radio(
        "Dilakukan lewat mana?", label_encoders["Channel"].classes_.tolist(),
        index=list(label_encoders["Channel"].classes_).index(fv["channel"]),
        horizontal=True,
        help="ATM = mesin ATM, Online = aplikasi/website, Branch = datang ke kantor cabang.",
    )
    occ = st.selectbox(
        "Pekerjaan nasabah", label_encoders["CustomerOccupation"].classes_.tolist(),
        index=list(label_encoders["CustomerOccupation"].classes_).index(fv["occ"]),
    )

st.write("")
predict_clicked = st.button("🔍 Tebak kelompok nasabah ini", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# Hasil prediksi
# ----------------------------------------------------------------------

if predict_clicked:
    input_dict = {
        "TransactionAmount": float(amount),
        "CustomerAge": int(age),
        "TransactionDuration": int(duration),
        "LoginAttempts": int(login),
        "AccountBalance": float(balance),
        "TransactionType_encoded": label_encoders["TransactionType"].transform([ttype])[0],
        "Channel_encoded": label_encoders["Channel"].transform([channel])[0],
        "CustomerOccupation_encoded": label_encoders["CustomerOccupation"].transform([occ])[0],
    }
    X_input = pd.DataFrame([input_dict])[feature_cols]
    X_scaled = clf_scaler.transform(X_input)

    pred_cluster = int(classifier.predict(X_scaled)[0])
    info = SEGMENT_INFO.get(pred_cluster, {
        "nama": f"Kelompok {pred_cluster}", "emoji": "❔", "warna": "#888",
        "ringkas": "", "detail": "", "saran": "",
    })

    confidence = None
    if hasattr(classifier, "predict_proba"):
        confidence = float(max(classifier.predict_proba(X_scaled)[0])) * 100

    st.divider()
    st.markdown(
        f"""
<div style="border:2px solid {info['warna']}22; border-left:6px solid {info['warna']};
            border-radius:12px; padding:20px 24px; background:{info['warna']}10;">
    <div style="font-size:38px; line-height:1;">{info['emoji']}</div>
    <div style="font-size:22px; font-weight:600; color:{info['warna']}; margin-top:6px;">
        {info['nama']}
    </div>
    <div style="font-size:15px; color:var(--text-secondary, #555); margin-top:4px;">
        {info['ringkas']}
    </div>
</div>
        """,
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
        st.caption(
            f"Model yang dipakai: {metadata['best_model_name']} · "
            f"Akurasi pada data pengujian: {metadata['best_model_accuracy']*100:.2f}%"
        )
else:
    st.caption("Isi data di atas lalu klik tombol untuk melihat hasilnya.")

st.divider()
st.caption(
    "Catatan: hasil prediksi ini berdasarkan pola dari data transaksi historis, "
    "sifatnya sebagai bahan pertimbangan — bukan keputusan akhir."
)