import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import joblib

# Judul halaman
st.markdown("<h1 style='text-align: center; border-bottom: 2px solid black; padding-bottom: 5px;'>PREDIKSI CHURN</h1>", unsafe_allow_html=True)

# Fitur di sidebar
st.sidebar.title('Home')
st.sidebar.title('Prediksi Churn')
st.sidebar.title('Profil')

# Fitur untuk memasukkan data
st.markdown("<h5 style='text-align: center; '>SILAHKAN MASUKAN DATA ANDA!</h5>", unsafe_allow_html=True)

# Input fitur-fitur
# Credit score
CreditScore = st.slider("Masukan Jumlah Credit Score Anda!", min_value=250, max_value=900)
# Domisili
geography = st.text_input('Domisili', '')
# Jenis kelamin
Gender = st.selectbox(
    'Masukan Jenis Kelamin Anda!',
    ('Laki-Laki', 'Perempuan'))
# Umur
Age = st.number_input("Berapa Umur Anda Saat Ini?", 0)
# Tenure
Tenure = st.number_input("Berapa lama Anda Telah Menjadi Nasabah? (masukan dalam bulanan)", 0)
# Saldo
Balance = st.text_input('Masukan Jumlah Saldo Anda!', '')
# Jumlah layanan bank
NumOfProduct = st.selectbox(
    'Berapa Jumlah Layanan Bank yang Anda Miliki?',
    ('1', '2', '3', '4', '>4'))
# Kartu kredit
HasCrCard = st.selectbox(
    'Apakah Anda Memiliki Kartu Credit?',
    ('Ya', 'Tidak'))
# Aktif sebagai member
IsActiveMember = st.selectbox(
    'Apakah Anda Adalah Pengguna Aktif?',
    ('Ya', 'Tidak'))
# Gaji tahunan
EstimatedSalary = st.selectbox(
    'Berapa Perkiraan Gaji Tahunan Anda?',
    ('0', '<10.000.000', '10.000.000 - 20.000.000', '20.000.000 - 30.000.000','>30.000.000'))
# Kemungkinan churn
Exited = st.selectbox(
    'Apakah Anda Berkemungkinan untuk Churn?',
    ('Ya', 'Tidak'))

# Load model yang telah disimpan sebelumnya
model_filename = 'random_forest_model.joblib'
loaded_model = joblib.load(model_filename)

# Ketika tombol "Prediksi" ditekan
if st.button("Prediksi"):

    # Mengubah data pengguna menjadi format yang dapat diproses oleh model
    user_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProduct],
        'HasCrCard': [1 if HasCrCard == 'Ya' else 0],
        'IsActiveMember': [1 if IsActiveMember == 'Ya' else 0],
        'EstimatedSalary': [EstimatedSalary],
        'Exited': [1 if Exited == 'Ya' else 0]
    })

    # Prediksi hasil Status
    hasil_prediksi = loaded_model.predict(user_data)

    # Mapping hasil prediksi ke label yang sesuai
    if hasil_prediksi == 0:
        status = "Tidak Churn"
    elif hasil_prediksi == 1:
        status = "Churn"

    # Menampilkan hasil prediksi
    st.write(f"Hasil Prediksi Churn Anda adalah: {status}")