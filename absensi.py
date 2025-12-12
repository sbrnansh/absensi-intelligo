import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import datetime
import os
import cv2

# Definisikan layer kustom untuk menangani parameter 'groups'
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Hapus parameter 'groups' jika ada (tidak dikenali oleh Keras DepthwiseConv2D)
        kwargs.pop('groups', None)
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

# Muat model dengan custom objects
try:
    model = tf.keras.models.load_model('keras_model.h5', custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Muat label
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Fungsi prediksi
def predict(image):
    image = image.resize((224, 224))  # Ukuran default Teachable Machine
    img_array = np.asarray(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    return labels[idx], prediction[0][idx]

# Setup file absensi
absen_file = "absensi.csv"
if not os.path.exists(absen_file):
    pd.DataFrame(columns=["Nama", "Waktu"]).to_csv(absen_file, index=False)

# Judul aplikasi
st.title("📸 Absensi Otomatis Siswa - Teachable Machine")

# Pilih metode input
option = st.radio("Pilih Metode Input:", ["📁 Upload Foto", "📷 Gunakan Webcam"])

image = None

# Opsi upload foto
if option == "📁 Upload Foto":
    uploaded_file = st.file_uploader("Upload Foto Wajah", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Foto yang diupload")

# Opsi webcam
elif option == "📷 Gunakan Webcam":
    img_file_buffer = st.camera_input("Ambil Foto via Webcam")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Foto dari Webcam")

# Proses gambar jika ada
if image is not None:
    name, conf = predict(image)

    if conf > 0.85:  # Threshold kepercayaan
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.read_csv(absen_file)

        # Cek apakah sudah absen hari ini
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        already = df[(df['Nama'] == name) & (df['Waktu'].str.contains(today))]

        if already.empty:
            new_row = pd.DataFrame([[name, now]], columns=["Nama", "Waktu"])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(absen_file, index=False)
            st.success(f"{name} berhasil absen ✅")
        else:
            st.warning(f"{name} sudah absen hari ini ⏱")
    else:
        st.error("Wajah tidak dikenali ❌")

# Tampilkan data absensi jika dipilih
if st.checkbox("📄 Lihat Absensi"):
    st.dataframe(pd.read_csv(absen_file))