import streamlit as st
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle
import matplotlib.pyplot as plt




st.sidebar.write("Proyek Sains Data")
st.sidebar.write("Harga Bawang merah Kabupaten Sumenep")
st.sidebar.write("Wahyu Rohmatul Abidin")
st.sidebar.write("22041100198")

with open('bagging_model2.pkl', 'rb') as file:
    bagging_model2 = pickle.load(file)

st.write("==================================================================")


st.title("Prediksi Harga Mingguan")
st.write("Masukkan data 3 minggu sebelumnya:")

week1 = st.number_input("Minggu 1", value=0.0)
week2 = st.number_input("Minggu 2", value=0.0)
week3 = st.number_input("Minggu 3", value=0.0)



st.write("==================================================================")

if st.button("Prediksi"):
    # Membentuk data input untuk prediksi
    input_data = np.array([week1, week2, week3]).reshape(1, -1)

    # Prediksi dengan model
    prediction = bagging_model2.predict(input_data)

    # Menampilkan hasil prediksi
    st.write(f"Prediksi harga untuk minggu ke-4 adalah: {prediction[0]:.2f}")




    st.write("Grafik prediksi harga:")
    fig, ax = plt.subplots()
    weeks = ['Minggu 1', 'Minggu 2', 'Minggu 3', 'Prediksi Minggu 4']
    values = [week1, week2, week3, prediction[0]]

    ax.plot(weeks, values, marker='o')
    ax.set_ylabel("Harga")
    ax.set_title("Prediksi Harga Mingguan")
    st.pyplot(fig)
