import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

# Sidebar Menu
menu = st.sidebar.radio("Pilih Menu", ["Prediksi K-NN", "Visualisasi Data", "Rekomendasi Mobil"])

# Fungsi untuk mengkategorikan penjualan
def categorize_sales(x):
    if x > 450:
        return 'sangat laris'
    elif x > 76:
        return 'cukup laris'
    else:
        return 'kurang laris'

# Load Data Train
@st.cache_data
def load_data():
    df_train = pd.read_excel("train_final.xlsx")  # Ganti sesuai nama file train
    df_test = pd.read_excel("test_final.xlsx")  # Ganti sesuai nama file test
    df_train.rename(columns={'Trans': 'trans'}, inplace=True)
    # Terapkan fungsi categorize_sales pada kolom 'Jumlah' untuk kedua dataset
    df_train['kategori_penjualan'] = df_train['Jumlah'].apply(categorize_sales)
    df_test['kategori_penjualan'] = df_test['Jumlah'].apply(categorize_sales)
    return df_train, df_test

df_train, df_test = load_data()

if menu == "Prediksi K-NN":
    st.title("🔍 Prediksi K-NN")

    # Sidebar: Informasi kategori mobil
    st.sidebar.markdown("### 📌 Keterangan Kategori Mobil")
    st.sidebar.write("Kategori label angka ini untuk input data prediksi dan rekomendasi mobil:")
    st.sidebar.write("""
    **Kategori Mobil:**
    - 4X2 TYPE SALES → 0  
    - 4X4 TYPE SALES → 1  
    - DOUBLE CABIN 4X2 / 4X4 → 2  
    - PICK UP → 3  
    - Sedan → 4  
    - TRUCK → 5  

    **Transmisi (Trans):**
    - AT → 0  
    - CVT → 1  
    - MT → 2  

    **Jenis Bahan Bakar (Fuel):**
    - BEV → 0  
    - D → 1  
    - EV → 2  
    - G → 3  
    - HEV → 4  
    - HYBRID → 5  
    
    **Kategori Penjualan:**
    - cukup laris → 0  
    - kurang laris → 1  
    - sangat laris → 2  
    """)

    # Sidebar: Pilih Fitur dan Target
    st.sidebar.header("Pengaturan Model K-NN")
    st.sidebar.write("**Input Prediksi Berdasarkan Fitur Spesifikasi Mobil:**")

    # Modifikasi kolom fitur menjadi sesuai permintaan
    selected_features = st.sidebar.multiselect(
        "Pilih Fitur", 
        ['Kategori Mobil', 'CC', 'Tank Capt', 'PS/HP', 'SEATER', 'Harga', 'trans', 'fuel']  # Fitur yang diminta
    )
    target = st.sidebar.selectbox("Pilih Target", ['kategori_penjualan'])  # Target tetap sama
    k_neighbors = st.sidebar.slider("Jumlah K", min_value=1, max_value=min(20, len(df_train)-1), value=5, step=1)

    if selected_features and target:
        # Data Latih
        X_train = df_train[selected_features]
        y_train = df_train[target]

        # Data Uji (Test Data)
        X_test = df_test[selected_features]
        y_test = df_test[target]

        # Normalisasi Data Train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # Terapkan normalisasi yang sama pada data uji

        # Model K-NN
        model = KNeighborsClassifier(n_neighbors=k_neighbors)
        model.fit(X_train_scaled, y_train.values.ravel())  # Melatih model dengan data latih

        # **Input Data Uji dari User**
        st.header("Masukkan Data Uji")
        input_data = []
        for feature in selected_features:
            value = st.number_input(f"Masukkan nilai untuk {feature}", value=float(df_train[feature].mean()))
            input_data.append(value)

        if st.button("Prediksi"):
            input_array = np.array(input_data).reshape(1, -1)  # Ubah ke array 2D
            input_scaled = scaler.transform(input_array)  # Normalisasi sesuai data train
            prediction = model.predict(input_scaled)[0]

            st.success(f"Prediksi: {prediction}")

    else:
        st.warning("Silakan pilih fitur dan target terlebih dahulu.")

elif menu == "Visualisasi Data":
    st.title("📊 Visualisasi Data")

    # Menampilkan Data
    st.subheader("DATA KLASIFIKASI TRAIN")
    st.dataframe(df_train)  # Menampilkan data dalam bentuk tabel interaktif

    st.subheader("DATA KLASIFIKASI TEST")
    st.dataframe(df_test)  # Menampilkan data dalam bentuk tabel interaktif

    # Elbow Method
    st.subheader("Elbow Method for KNN")

    # Load data elbow
    @st.cache_data
    def load_data_elbow():
        yh = pd.read_excel('bersih train fix.xlsx')
        yt = pd.read_excel('bersih test fix.xlsx')
        return yh, yt

    yh, yt = load_data_elbow()

    # Define features and target
    features = ['Kategori Mobil', 'CC', 'Tank Capt', 'PS/HP', 'SEATER', 'trans', 'fuel', 'Harga']
    X_train = yh[features]
    Y_train = yh['kategori_penjualan']
    X_test = yt[features]
    Y_test = yt['kategori_penjualan']

    # Normalize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Elbow method for optimal k
    k_range = range(1, 21)
    scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X_train_scaled, Y_train, cv=5, scoring='accuracy').mean()
        scores.append(score)

    # Plot interactive elbow method using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(k_range), y=scores, mode='lines+markers', marker=dict(size=8), line=dict(dash='dash')))

    # Update layout for better visualization
    fig.update_layout(
        title="Elbow Method for Optimal k in KNN",
        xaxis_title="Number of Neighbors (k)",
        yaxis_title="Cross-Validated Accuracy",
        template="plotly_white"
    )

    # Show interactive chart
    st.plotly_chart(fig, use_container_width=True)

    # Model K-NN untuk Prediksi Actual vs Predicted
    knn = KNeighborsClassifier(n_neighbors=5)  # Menggunakan k=5 sebagai contoh
    knn.fit(X_train_scaled, Y_train)

    # Predict values
    Y_pred_knn = knn.predict(X_test_scaled)

    # Line Plot
    st.subheader("Actual vs Predicted Kategori Penjualan")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(Y_test))),
        y=Y_test,
        mode='lines+markers',
        name='Actual',
        line=dict(dash='solid', color='blue'),
        marker=dict(symbol='circle', size=6)
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(Y_pred_knn))),
        y=Y_pred_knn,
        mode='lines+markers',
        name='Predicted',
        line=dict(dash='dash', color='red'),
        marker=dict(symbol='x', size=6)
    ))

    fig.update_layout(
        title="Actual vs Predicted Kategori Penjualan",
        xaxis_title="Data Point",
        yaxis_title="Kategori Penjualan",
        template="plotly_white",
        legend_title="Legend",
        hovermode="x unified"
    )

    st.plotly_chart(fig)

    # **Feature Importance Calculation and Plotting** 
    st.subheader("Feature Importance from Permutation Importance")

    # Calculate permutation importance
    result = permutation_importance(
        knn, X_test_scaled, Y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    # Get feature importances
    importances = result.importances_mean

    # Create a DataFrame for better visualization
    feature_importances = pd.DataFrame({'feature': features, 'importance': importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    # Plot the feature importances using Plotly for interactivity
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=feature_importances['feature'],
        y=feature_importances['importance'],
        name='Importance',
        marker=dict(color='royalblue')
    ))

    fig.update_layout(
        title='Feature Importance from Permutation Importance',
        xaxis_title='Feature',
        yaxis_title='Importance',
        xaxis_tickangle=-45,  # Rotate x-axis labels for readability
        template='plotly_dark'
    )

    # Display the interactive plot
    st.plotly_chart(fig)

elif menu == "Rekomendasi Mobil":
    st.title("🚗 Rekomendasi Mobil")

    # Load dataset
    @st.cache_data
    def load_data_jarak():
        return pd.read_excel("jarak.xlsx")  # Sesuaikan dengan lokasi file

    # Load data
    df = load_data_jarak()

    # Sidebar untuk input spesifikasi mobil
    st.sidebar.header("Input Spesifikasi Mobil")

    # Kategori Mobil: Tambahkan opsi "Semua Kategori"
    kategori_mobil = st.sidebar.selectbox("Kategori Mobil", ["Semua Kategori"] + df["Kategori Mobil"].unique().tolist())

    # Brand Mobil: Tambahkan opsi "Semua Brand"
    brand_mobil = st.sidebar.selectbox("Brand Mobil", ["Semua Brand"] + df["Brand"].unique().tolist())

    # Input spesifikasi opsional
    cc_input = st.sidebar.number_input("Masukkan CC (Opsional)", min_value=0, value=0)
    tank_input = st.sidebar.number_input("Masukkan Tank Capt (Opsional)", min_value=0, value=0)
    ps_hp_input = st.sidebar.number_input("Masukkan PS/HP (Opsional)", min_value=0, value=0)
    seater_input = st.sidebar.number_input("Masukkan Seater (Opsional)", min_value=0, value=0)

    # Input filter harga (opsional: batas minimum dan maksimum)
    harga_min = st.sidebar.number_input("Masukkan Harga Minimum (Opsional)", min_value=0, value=0)
    harga_max = st.sidebar.number_input("Masukkan Harga Maksimum (Opsional)", min_value=0, value=0)

    # Filter data berdasarkan input pengguna
    filtered_df = df.copy()  # Mulai dari semua data

    # Terapkan filter kategori mobil jika pengguna memilih kategori tertentu
    if kategori_mobil != "Semua Kategori":
        filtered_df = filtered_df[filtered_df["Kategori Mobil"] == kategori_mobil]

    # Terapkan filter brand mobil jika pengguna memilih brand tertentu
    if brand_mobil != "Semua Brand":
        filtered_df = filtered_df[filtered_df["Brand"] == brand_mobil]

    # Terapkan filter spesifikasi hanya jika pengguna menginput nilai
    if cc_input > 0:
        filtered_df = filtered_df[filtered_df["CC"] == cc_input]
    if tank_input > 0:
        filtered_df = filtered_df[filtered_df["Tank Capt"] == tank_input]
    if ps_hp_input > 0:
        filtered_df = filtered_df[filtered_df["PS/HP"] == ps_hp_input]
    if seater_input > 0:
        filtered_df = filtered_df[filtered_df["SEATER"] == seater_input]

    # Terapkan filter harga jika nilai minimum atau maksimum diisi
    if harga_min > 0:
        filtered_df = filtered_df[filtered_df["Harga"] >= harga_min]
    if harga_max > 0 and harga_max >= harga_min:
        filtered_df = filtered_df[filtered_df["Harga"] <= harga_max]

    # Urutkan berdasarkan kolom "distance_to_s1_first_row" (jika ada)
    if "distance_to_s1_first_row" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by="distance_to_s1_first_row", ascending=True)

    # Tampilkan hasil rekomendasi tanpa kolom "distance_to_s1_first_row"
    if not filtered_df.empty:
        st.write("### Rekomendasi Mobil Berdasarkan Spesifikasi")
        st.dataframe(filtered_df.drop(columns=["distance_to_s1_first_row"], errors="ignore"))  # Sembunyikan kolom jarak
    else:
        st.write("Tidak ada mobil yang cocok dengan spesifikasi yang dimasukkan.")
