import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import tempfile
import os
from keras.preprocessing import image
from keras.utils import img_to_array

# Sidebar Navigation
menu = st.sidebar.selectbox(
    "Pilih Halaman",
    ("Beranda", "Preprocessing", "Pelatihan Model", "Evaluasi Model", "Prediksi", "Tentang Penelitian")
)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])

# ===================== BERANDA =====================
if menu == "Beranda":
    st.title("Implementasi Teknik Oversampling pada Klasifikasi Kanker Kulit")

    st.markdown("""
    ### Latar Belakang
    Kanker kulit merupakan salah satu jenis kanker yang umum terjadi di Asia, yang disebabkan oleh paparan sinar ultraviolet berlebih dari matahari sehingga mendorong pertumbuhan sel secara abnormal. Deteksi dini yang akurat sangat penting untuk mencegah penyebaran kanker lebih lanjut. Namun, proses diagnosis yang dilakukan oleh tenaga medis sering kali bersifat manual, sehingga berisiko menimbulkan kesalahan deteksi.

    Model klasifikasi berbasis komputer yang andal sangat dibutuhkan untuk membantu mengidentifikasi berbagai jenis kanker kulit secara otomatis. Salah satu metode yang terbukti efektif adalah **Convolutional Neural Network (CNN)** karena mampu mengenali pola-pola sel kanker kulit melalui citra dermoskopi, yang mengandung informasi penting seperti warna, tekstur, dan ukuran.

    Meskipun demikian, penerapan CNN pada dataset dermoskopi seperti **HAM10000** menghadapi tantangan berupa **ketidakseimbangan data**, yang menyebabkan model cenderung mengenali hanya kelas mayoritas. Untuk mengatasi masalah tersebut, teknik **Random Oversampling (ROS)** dapat digunakan sebagai solusi penyeimbang data.

    Di sisi lain, pendekatan **transfer learning** dengan memanfaatkan model pre-trained seperti **MobileNetV2** yang sudah dilatih pada dataset besar (ImageNet) telah terbukti mampu meningkatkan akurasi pada berbagai tugas klasifikasi citra, termasuk kanker kulit.
    """)

    st.markdown("### Pertanyaan Penelitian")
    st.markdown("""
    1. Bagaimana pengaruh penerapan teknik ROS untuk menyelesaikan permasalahan pada dataset HAM10000 yang tidak seimbang?
    2. Bagaimana pengaruh penerapan transfer learning ImageNet pada model klasifikasi yang sedang dikembangkan dengan menggunakan model pre-trained MobileNetV2 untuk dapat menghasilkan model klasifikasi citra kanker kulit terbaik?
    """)

    st.markdown("### Tujuan Penelitian")
    st.markdown("""
    1. Menganalisis pengaruh dari penerapan teknik ROS pada dataset HAM10000 yang akan digunakan pada proses klasifikasi sebagai bentuk solusi dari permasalahan pada dataset yang tidak seimbang.
    2. Menganalisis pengaruh dari penerapan transfer learning ImageNet pada model klasifikasi yang sedang dikembangkan dengan menggunakan model pre-trained MobileNetV2 untuk dapat menghasilkan model klasifikasi citra kanker kulit terbaik.
    """)

    st.markdown("### Batasan Masalah")
    st.markdown("""
    - Dataset yang digunakan adalah **HAM10000** dari Kaggle ([link dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)), terdiri dari **10.015 citra dermoskopi** yang terbagi dalam **7 kelas kanker kulit**.
    - Untuk menyeimbangkan dataset, digunakan teknik **Random Oversampling (ROS)**.
    - **Hyperparameter** yang digunakan adalah: learning rate = 0.001, batch size = 32, epoch = 50, optimizer = SGD (momentum = 0.9).
    - Evaluasi model menggunakan **5-fold cross-validation** pada data latih untuk menguji generalisasi model.
    """)

    st.markdown("### Perancangan Sistem")
    st.image("/asset/perancangan-sistem.png", caption="Diagram Perancangan Sistem", use_container_width=True)

    st.markdown("### Skenario Uji Coba")

    skenario_df = pd.DataFrame({
        "Skenario": [1, 2, 3, 4],
        "Random oversampling": ["Pakai", "Tidak Pakai", "Pakai", "Tidak Pakai"],
        "Transfer learning": ["ImageNet (fine-tuning)", "ImageNet (fine-tuning)", "Tanpa", "Tanpa"]
    })
    st.table(skenario_df)

    st.markdown("**Metode Klasifikasi:** MobileNetV2")
    st.markdown("**Hyper Parameter:**")
    st.markdown("""
    - Learning Rate: 0.001
    - Optimizer: SGD
    - Momentum: 0.9
    - Batch size: 32
    - Epoch: 50
    - Callback: ModelCheckpoint & EarlyStopping
    - Loss: sparse_categorical_crossentropy
    """)



# ===================== PREPROCESSING =====================
elif menu == "Preprocessing":

    st.title("Preprocessing Data")

    st.markdown("""
    ### Tahapan Preprocessing
    1. Resize citra dari ukuran awal ke 64x64 piksel.
    2. Transformasi ke bentuk numerik (vektor) agar dapat diterapkan Random Oversampling (ROS).
    3. Transformasi Label ke bentuk numerik menggunakan LabelEncoder
    4. Oversampling dilakukan pada data latih.
    5. De-transformasi ke bentuk citra kembali untuk pelatihan CNN.
    6. Augmentasi dan normalisasi menggunakan `ImageDataGenerator`.

    ----
    """)

    # ========== Sub Preprocessing - Resize Citra ==============
    st.subheader("1. Tahap Resize Citra")

    st.markdown("""
    Sebelum data citra digunakan untuk pelatihan, ukuran citra awal perlu diseragamkan.
    Citra dermoskopi dari dataset HAM10000 memiliki ukuran awal **650 x 450 piksel**.

    Berikut adalah simulasi resize citra menjadi **64 x 64 piksel**, seperti yang dilakukan dalam proses preprocessing dataset:

    #### Kode Fungsi Resize:
    ```python
    from keras.preprocessing import image
    from keras.utils import img_to_array

    def resize_image(img_path, target_size=(64, 64)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        return img_array
    ```
    """)

    def resize_image(img_path, target_size=(64, 64)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        return img_array

    # Gambar contoh harus ada di direktori yang sama
    example_img_path = "/asset/gambarpreprocessing.jpg"
    if os.path.exists(example_img_path):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Ukuran Asli (650 x 450)**")
            st.image(example_img_path, width=300)

        with col2:
            resized_image = resize_image(example_img_path)
            st.markdown("**Hasil Resize (64 x 64)**")
            st.image(resized_image.astype("uint8"), width=300)
    else:
        st.warning("Gambar contoh tidak ditemukan. Pastikan file `contoh_gambar.jpg` berada di direktori yang sama.")

    st.markdown("---")
    st.info("Lanjut ke proses transformasi data setelah resize...")

    # ========== Sub Preprocessing - Data Transformasi ==============
    st.subheader("2. Tahap Transformasi Data")

    st.markdown("""
    Citra hasil resize masih dalam bentuk array 3 dimensi (3 channel: RGB). Untuk dapat dilakukan proses oversampling, data citra perlu diubah ke dalam bentuk **vektor 1 dimensi** atau **flat array**.

    Metode yang digunakan dalam penelitian ini untuk transformasi citra adalah **Row Major Order**, yaitu pembacaan array baris per baris secara berurutan.

    #### Kode Fungsi Transformasi:
    ```python
    def transform_image(img_array):
        R = img_array[:, :, 0].flatten()
        G = img_array[:, :, 1].flatten()
        B = img_array[:, :, 2].flatten()
        reordered_array = np.concatenate([R, G, B])
        return reordered_array
    ```
    """)

    # Fungsi transformasi ke 1D menggunakan Row Major
    def transform_image(img_array):
        R = img_array[:, :, 0].flatten()
        G = img_array[:, :, 1].flatten()
        B = img_array[:, :, 2].flatten()
        reordered_array = np.concatenate([R, G, B])
        return R, G, B, reordered_array

    if os.path.exists(example_img_path):
        st.markdown("##### Gambar Hasil Resize:")
        st.image(resized_image.astype("uint8"), width=200)
        st.caption(f"Ukuran: {resized_image.shape}")  # (64, 64, 3)

        R, G, B, transformed = transform_image(resized_image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Channel Merah (R):**")
            red_only = np.zeros_like(resized_image)
            red_only[:, :, 0] = resized_image[:, :, 0] * 255
            st.image(red_only.astype("uint8"), caption="Merah (dalam RGB)", use_container_width=True)
            st.caption(f"Ukuran: {resized_image[:, :, 0].shape}")  # (64, 64)
            st.code(f"{R[:10]} ...")

        with col2:
            st.markdown("**Channel Hijau (G):**")
            green_only = np.zeros_like(resized_image)
            green_only[:, :, 1] = resized_image[:, :, 1] * 255
            st.image(green_only.astype("uint8"), caption="Hijau (dalam RGB)", use_container_width=True)
            st.caption(f"Ukuran: {resized_image[:, :, 1].shape}")
            st.code(f"{G[:10]} ...")

        with col3:
            st.markdown("**Channel Biru (B):**")
            blue_only = np.zeros_like(resized_image)
            blue_only[:, :, 2] = resized_image[:, :, 2] * 255
            st.image(blue_only.astype("uint8"), caption="Biru (dalam RGB)", use_container_width=True)
            st.caption(f"Ukuran: {resized_image[:, :, 2].shape}")
            st.code(f"{B[:10]} ...")

        st.markdown("##### Vektor 1D Hasil Transformasi (R → G → B):")
        st.caption(f"Ukuran: {transformed.shape}")  # (12288,)
        st.code(f"{transformed[:10]} ... {transformed[len(R):len(R)+10]} ... {transformed[-10:]}")

    else:
        st.warning("Transformasi tidak dapat dilakukan karena gambar tidak ditemukan.")

    # ========== Sub Preprocessing - LabelEncoder ==============
    st.subheader("3. Tahap Label Encoding")

    st.markdown("""
    Sebelum citra dipasangkan dengan label untuk keperluan pelatihan, label kategori seperti `akiec`, `bcc`, `mel`, dll perlu diubah ke format numerik agar bisa diproses oleh model klasifikasi.

    Untuk itu digunakan **`LabelEncoder`** dari `sklearn`, yang secara otomatis mengubah label string menjadi angka.

    #### Tujuan Penggunaan LabelEncoder:
    - Mengubah data kategorikal menjadi numerik agar dapat digunakan oleh algoritma klasifikasi.
    - Memastikan bahwa setiap kelas memiliki representasi numerik unik (misal: `akiec` → 0, `bcc` → 1, dst).

    #### Kode yang Digunakan:
    ```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    ```
    """)

    st.markdown("##### Contoh Label:")
    label_kategori = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_numerik = label_encoder.fit_transform(label_kategori)

    label_df = pd.DataFrame({
        "Label Kategorikal": label_kategori,
        "Label Numerik": label_numerik
    })

    st.dataframe(label_df, use_container_width=True)

    # ========== Sub Preprocessing - Penerapan Random Oversampling pada Data Train ==============

    st.subheader("4. Tahap Balancing Data dengan Random Oversampling")

    st.markdown("""
    Pada tahap ini dilakukan proses **balancing dataset** menggunakan teknik **Random Oversampling (ROS)** pada data citra yang sebelumnya telah diubah menjadi bentuk numerik (1 dimensi) melalui proses transformasi.

    Tujuan dari ROS adalah untuk meningkatkan jumlah sampel pada **kelas minoritas** dengan cara menduplikasi secara acak data dari kelas tersebut, hingga jumlahnya setara dengan kelas mayoritas.

    Berikut adalah diagram alur proses ROS dalam penelitian ini:
    """)

    st.image("/asset/alurros.png", caption="Alur Proses Random Oversampling", use_container_width=True)

    st.markdown("""
    #### Langkah-langkah Proses ROS:
    - Masukkan data train yang telah di-transformasi.
    - Tentukan jumlah data per kelas untuk mengetahui mana yang mayoritas dan minoritas.
    - Hitung selisih antara jumlah data kelas mayoritas dan minoritas.
    - Untuk setiap kelas minoritas:
      - Duplikat acak data hingga jumlahnya sama dengan mayoritas.
    - Gabungkan seluruh data yang telah diseimbangkan.

    #### Kode Implementasi Manual Random Oversampling:
    ```python
    from collections import Counter
    from sklearn.utils import resample
    import numpy as np

    # Fungsi manual Random Oversampling
    def random_oversample(X, y):
        classes = np.unique(y)
        counts = Counter(y)
        max_count = max(counts.values())

        X_resampled = []
        y_resampled = []

        for label in classes:
            idx = np.where(y == label)[0]
            X_label = X[idx]

            # Oversampling data fitur
            X_upsampled = resample(X_label,
                                   replace=True,
                                   n_samples=max_count,
                                   random_state=42)

            # Buat label yang sesuai jumlah oversampling
            y_upsampled = np.array([label] * max_count)

            X_resampled.append(X_upsampled)
            y_resampled.append(y_upsampled)

        X_final = np.vstack(X_resampled)
        y_final = np.hstack(y_resampled)
        return X_final, y_final
    ```

    ### Perbandingan Sebelum dan Sesudah ROS
    Di bawah ini adalah visualisasi perbandingan jumlah data train berdasarkan label sebelum dan sesudah dilakukan balancing:

    """)

    st.image("/asset/hasilros.png", caption="Perbandingan Data Train Sebelum dan Sesudah ROS", use_container_width=True)

    # ========== Sub Preprocessing - Data De-Transformasi ==============

    st.subheader("5. Tahap De-transformasi")

    st.markdown("""
    Setelah dilakukan proses oversampling, data hasil transformasi berada dalam bentuk **vektor 1 dimensi (1×12288)**. Agar dapat digunakan oleh model CNN (yang memerlukan input citra RGB), maka data tersebut perlu dikembalikan ke bentuk semula, yaitu **array citra 64×64×3**.

    #### Tujuan De-transformasi:
    - Mengembalikan struktur array ke bentuk citra 3 dimensi.
    - Memastikan format input sesuai untuk digunakan dalam pelatihan model klasifikasi.

    #### Kode yang Digunakan:
    ```python
    # Data latih yang sudah setara akan dilakukan de-transformasi ke bentuk 64x64x3
    X_train_reshape = np.array(X_train_ros).reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
    ```
    """)

    # Simulasi sederhana jika pakai data 1 baris dari 'transformed'
    st.markdown("##### Contoh De-transformasi:")
    try:
        st.markdown("##### Vektor 1D Hasil Transformasi (R → G → B):")
        st.caption(f"Ukuran: {transformed.shape}")  # (12288,)
        st.code(f"{transformed[:10]} ... {transformed[len(R):len(R)+10]} ... {transformed[-10:]}")


        st.markdown("##### Vektor 3D Hasil De-Transformasi (R → G → B):")
        reshaped_example = transformed.reshape(3, 64, 64).transpose(1, 2, 0)
        st.image(reshaped_example.astype("uint8"), caption="Hasil De-transformasi (64x64x3)", width=200)
        st.caption(f"Ukuran: {reshaped_example.shape}")
    except:
        st.warning("Gagal melakukan reshape. Pastikan data berbentuk 12288 (64x64x3).")


    # ========== Sub Preprocessing - Augmentasi dan normalisasi menggunakan `ImageDataGenerator` ==============
    st.subheader("6. Tahap Augmentasi dan Normalisasi")

    st.markdown("""
    Tahap akhir preprocessing adalah augmentasi dan normalisasi data citra sebelum digunakan dalam proses pelatihan model.

    #### Tujuan:
    - **Augmentasi** dilakukan untuk memperbesar jumlah variasi citra pelatihan dengan memanipulasi citra (rotasi, pergeseran, zoom, dll), sehingga model lebih robust dan tidak mudah overfitting.
    - **Normalisasi** dilakukan untuk merubah nilai piksel dari rentang [0–255] ke [0–1], agar proses pelatihan lebih stabil dan cepat konvergen.

    #### Kapan digunakan:
    - Augmentasi hanya diterapkan pada **data latih**
    - Normalisasi diterapkan pada **data latih dan validasi**

    #### Konfigurasi Augmentasi:
    """)

    # Tabel konfigurasi augmentasi
    aug_config = {
        "Teknik Augmentasi": ["rotation_range", "width_shift_range", "height_shift_range", "zoom_range"],
        "Nilai": ["10", "0.1", "0.1", "0.1"]
    }
    st.table(pd.DataFrame(aug_config))

    st.markdown("#### Kode Implementasi:")
    st.code("""
    # Membuat fungsi ImageDataGenerator untuk data train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    # Membuat fungsi ImageDataGenerator untuk data validasi
    val_datagen = ImageDataGenerator(rescale=1./255)
        """)

    st.markdown("### Simulasi Augmentasi dan Normalisasi")

    # Gambar dasar untuk augmentasi
    if os.path.exists(example_img_path):
        st.markdown("##### Gambar Asli:")
        original_img = Image.open(example_img_path).convert("RGB").resize((64, 64))
        st.image(original_img, caption="Gambar Asli (64x64)", width=150)

        # Ubah ke array dan expand dims
        img_array = img_to_array(original_img)
        img_array_expanded = np.expand_dims(img_array, axis=0)

        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        st.markdown("### Simulasi Dampak Masing-Masing Teknik Augmentasi")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Rotasi 10°**")
            rot_gen = ImageDataGenerator(rotation_range=10)
            rot_img = next(rot_gen.flow(img_array_expanded, batch_size=1))[0].astype("float32") / 255.0
            st.image(rot_img, caption="Rotasi", use_container_width=True)

        with col2:
            st.markdown("**Geser Horizontal 10%**")
            shift_w_gen = ImageDataGenerator(width_shift_range=0.1)
            shift_w_img = next(shift_w_gen.flow(img_array_expanded, batch_size=1))[0].astype("float32") / 255.0
            st.image(shift_w_img, caption="Geser X", use_container_width=True)

        with col3:
            st.markdown("**Geser Vertikal 10%**")
            shift_h_gen = ImageDataGenerator(height_shift_range=0.1)
            shift_h_img = next(shift_h_gen.flow(img_array_expanded, batch_size=1))[0].astype("float32") / 255.0
            st.image(shift_h_img, caption="Geser Y", use_container_width=True)

        with col4:
            st.markdown("**Zoom 10%**")
            zoom_gen = ImageDataGenerator(zoom_range=0.1)
            zoom_img = next(zoom_gen.flow(img_array_expanded, batch_size=1))[0].astype("float32") / 255.0
            st.image(zoom_img, caption="Zoom", use_container_width=True)

        # NORMALISASI
        st.markdown("### Perbandingan Nilai Piksel Sebelum dan Sesudah Normalisasi")
        st.write("Berikut ini adalah nilai piksel pada channel R (merah) sebelum dan sesudah dilakukan normalisasi:")

        # Ambil piksel sebelum normalisasi
        st.markdown("**Sebelum Normalisasi (Rentang 0–255):**")
        st.code(img_array[:, :, 0][:5, :5])  # channel R gambar asli (tanpa rescale)

        # NORMALISASI
        st.markdown("**Sesudah Normalisasi (Rentang 0–1):**")
        st.code(zoom_img[:, :, 0][:5, :5])  # channel R gambar augmentasi (sudah dinormalisasi)



    else:
        st.warning("Gambar contoh tidak ditemukan. Pastikan `contoh_gambar.jpg` tersedia.")



# ===================== PELATIHAN =====================
elif menu == "Pelatihan Model":
    st.title("Pelatihan Model Klasifikasi Citra Kanker Kulit")

    st.markdown("""
    Pada tahap ini dilakukan pelatihan model klasifikasi citra kanker kulit dengan dua pendekatan berbeda:

    1. **Model CNN Manual (Custom MobileNetV2)**: Dirancang dari awal dengan menyusun sendiri arsitektur seperti MobileNetV2.
    2. **Model Transfer Learning**: Menggunakan model MobileNetV2 yang telah dilatih pada dataset ImageNet, lalu dilakukan fine-tuning agar dapat mengenali citra kanker kulit.

    Untuk menjamin generalisasi model, proses pelatihan dilakukan menggunakan **5-Fold Cross Validation**.
    """)

    # --- CNN Manual ---
    st.subheader("1. Model CNN Manual (Custom MobileNetV2)")

    st.markdown("""
    Model ini dibangun dengan menyusun sendiri layer-layer konvolusi dan blok *inverted residual* seperti arsitektur MobileNetV2 asli, namun tanpa memanfaatkan bobot pre-trained. Model dilatih dari awal menggunakan data yang telah seimbang dan ditransformasikan.
    """)

    st.image("/asset/mobilenetv2manual.jpg", caption="Struktur Arsitektur CNN Manual", use_container_width=True)

    st.markdown("#### Tabel Arsitektur CNN Manual per Layer:")
    import pandas as pd

    tabel_manual = {
        "Layer": [
            "Input", "Conv1", "Inverted Residual Block 1", "Inverted Residual Block 2", "Inverted Residual Block 3",
            "Inverted Residual Block 4", "Inverted Residual Block 5", "Inverted Residual Block 6", "Inverted Residual Block 7",
            "Inverted Residual Block 8", "Inverted Residual Block 9", "Inverted Residual Block 10", "Inverted Residual Block 11",
            "Inverted Residual Block 12", "Inverted Residual Block 13", "Inverted Residual Block 14",
            "Inverted Residual Block 15", "Inverted Residual Block 16", "Inverted Residual Block 17",
            "Conv2", "GlobalAveragePooling2D", "Output Layer + Softmax"
        ],
        "Citra Input": [
            "64 x 64 x 3", "64 x 64 x 3", "32 x 32 x 32", "32 x 32 x 16", "16 x 16 x 24",
            "16 x 16 x 24", "8 x 8 x 32", "8 x 8 x 32", "8 x 8 x 32",
            "4 x 4 x 64", "4 x 4 x 64", "4 x 4 x 64", "4 x 4 x 64",
            "4 x 4 x 64", "4 x 4 x 96", "4 x 4 x 96",
            "2 x 2 x 160", "2 x 2 x 160", "2 x 2 x 160",
            "2 x 2 x 320", "2 x 2 x 1280", "1280"
        ],
        "Citra Output": [
            "64 x 64 x 3", "32 x 32 x 32", "32 x 32 x 16", "16 x 16 x 24", "16 x 16 x 24",
            "8 x 8 x 32", "8 x 8 x 32", "8 x 8 x 32", "4 x 4 x 64",
            "4 x 4 x 64", "4 x 4 x 64", "4 x 4 x 64", "4 x 4 x 96",
            "4 x 4 x 96", "4 x 4 x 96", "2 x 2 x 160",
            "2 x 2 x 160", "2 x 2 x 160", "2 x 2 x 320",
            "2 x 2 x 1280", "1280", "Jumlah Kelas"
        ],
        "Stride": [
            "-", "2", "1", "2", "1",
            "2", "1", "1", "2",
            "1", "1", "1", "1",
            "1", "1", "2",
            "1", "1", "1",
            "1", "-", "-"
        ],
        "Filter": [
            "-", "32", "16", "24", "24",
            "32", "32", "32", "64",
            "64", "64", "64", "96",
            "96", "96", "160",
            "160", "160", "320",
            "1280", "-", "-"
        ]
    }

    df_manual = pd.DataFrame(tabel_manual)
    st.dataframe(df_manual, use_container_width=True)

    st.markdown("#### Kode Arsitektur CNN Manual:")
    st.code("""
# Fungsi dan implementasi blok residual sudah dijelaskan dalam model
# Berikut hanya ringkasan fungsi utama
model = create_model()
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    """)

    # --- Transfer Learning ---
    st.subheader("2. Model Transfer Learning (MobileNetV2 Pre-trained)")

    st.markdown("""
    Model ini menggunakan arsitektur **MobileNetV2** yang telah dilatih sebelumnya pada dataset ImageNet. Teknik ini dikenal sebagai **Transfer Learning**. Semua layer dari model pre-trained dibuat **trainable** (fine-tuned) agar dapat menyesuaikan terhadap citra dermoskopi yang digunakan.
    """)

    st.image("/asset/pretrainedmobilenetv2.jpg", caption="Struktur Transfer Learning MobileNetV2", use_container_width=True)

    st.markdown("#### Kode Implementasi:")
    st.code("""
def create_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    base_model.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(7, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
    """)

    # --- K-Fold CV ---
    st.subheader("3. Evaluasi dengan 5-Fold Cross Validation")

    st.markdown("""
    Untuk memastikan model tidak hanya belajar dari subset data tertentu, proses pelatihan dilakukan menggunakan **5-Fold Cross Validation**. Data latih dibagi menjadi 5 bagian, model dilatih 5 kali dengan rotasi bagian data validasi, lalu hasilnya dirata-rata untuk menilai kinerja secara umum.

    Teknik ini meningkatkan keandalan model dan menghindari overfitting.
    """)

    st.markdown("#### Ilustrasi:")
    st.image("/asset/kfold_ilustrasi.PNG", caption="Skema 5-Fold Cross Validation", use_container_width=True)


# ===================== EVALUASI =====================
elif menu == "Evaluasi Model":
    st.title("Evaluasi Model Berdasarkan Skenario Uji Coba")

    st.markdown("""
    Evaluasi dilakukan terhadap empat skenario pengujian untuk menilai performa model klasifikasi. Setiap skenario mencerminkan kombinasi penggunaan **Random Oversampling (ROS)** dan **Transfer Learning**:

    1. ROS + Transfer Learning
    2. Tanpa ROS + Transfer Learning
    3. ROS + Tanpa Transfer Learning
    4. Tanpa ROS + Tanpa Transfer Learning

    Setiap skenario mencakup dua bagian evaluasi:
    - **Hasil Pelatihan** menggunakan 5-Fold Cross Validation
    - **Hasil Pengujian** model terbaik pada data uji
    """)

    # Data per skenario
    skenario_list = [
        {
            "judul": "Skenario Pertama (ROS + Transfer Learning)",
            "train_table": {
                "Fold": [5],
                "Best Epoch": [47],
                "Stop Epoch": [50],
                "Akurasi": ["99,52%"],
                "Loss": ["1,44%"],
                "Val Akurasi": ["98,60%"],
                "Val Loss": ["6,29%"]
            },
            "train_imgs": [
                "/asset/Skenario Pertama/Pelatihan/akurasi.png",
                "/asset/Skenario Pertama/Pelatihan/loss.png"
            ],
            "test_akurasi": "79%",
            "test_metrics": {
                "": ["precision", "recall", "f1-score"],
                "macro avg": [0.69, 0.69, 0.68],
                "weighted avg": [0.82, 0.79, 0.80]
            },
            "conf_matrix": "/asset/Skenario Pertama/Pengujian/confusionmatrix.png",
            "roc": "/asset/Skenario Pertama/Pengujian/kurvaroc.png"
        },
        {
            "judul": "Skenario Kedua (Tanpa ROS + Transfer Learning)",
            "train_table": {
                "Fold": [1],
                "Best Epoch": [29],
                "Stop Epoch": [39],
                "Akurasi": ["91,52%"],
                "Loss": ["22,8%"],
                "Val Akurasi": ["78,39%"],
                "Val Loss": ["90,12%"]
            },
            "train_imgs": [
                "/asset/Skenario Kedua/Pelatihan/akurasi.png",
                "/asset/Skenario Kedua/Pelatihan/loss.png"
            ],
            "test_akurasi": "77%",
            "test_metrics": {
                "": ["precision", "recall", "f1-score"],
                "macro avg": [0.65, 0.49, 0.54],
                "weighted avg": [0.76, 0.77, 0.75]
            },
            "conf_matrix": "/asset/Skenario Kedua/Pengujian/confusionmatrix.png",
            "roc": "/asset/Skenario Kedua/Pengujian/kurvaroc.png"
        },
        {
            "judul": "Skenario Ketiga (ROS + Tanpa Transfer Learning)",
            "train_table": {
                "Fold": [3],
                "Best Epoch": [50],
                "Stop Epoch": [50],
                "Akurasi": ["94,59%"],
                "Loss": ["15,3%"],
                "Val Akurasi": ["96,82%"],
                "Val Loss": ["10,1%"]
            },
            "train_imgs": [
                "/asset/Skenario Ketiga/Pelatihan/akurasi.png",
                "/asset/Skenario Ketiga/Pelatihan/loss.png"
            ],
            "test_akurasi": "69%",
            "test_metrics": {
                "": ["precision", "recall", "f1-score"],
                "macro avg": [0.46, 0.58, 0.51],
                "weighted avg": [0.75, 0.69, 0.71]
            },
            "conf_matrix": "/asset/Skenario Ketiga/Pengujian/confusionmatrix.png",
            "roc": "/asset/Skenario Ketiga/Pengujian/kurvaroc.png"
        },
        {
            "judul": "Skenario Keempat (Tanpa ROS + Tanpa Transfer Learning)",
            "train_table": {
                "Fold": [2],
                "Best Epoch": [29],
                "Stop Epoch": [39],
                "Akurasi": ["72,93%"],
                "Loss": ["73,6%"],
                "Val Akurasi": ["72,93%"],
                "Val Loss": ["80,40%"]
            },
            "train_imgs": [
                "/asset/Skenario Keempat/Pelatihan/akurasi.png",
                "/asset/Skenario Keempat/Pelatihan/loss.png"
            ],
            "test_akurasi": "71%",
            "test_metrics": {
                "": ["precision", "recall", "f1-score"],
                "macro avg": [0.39, 0.30, 0.32],
                "weighted avg": [0.67, 0.71, 0.68]
            },
            "conf_matrix": "/asset/Skenario Keempat/Pengujian/confusionmatrix.png",
            "roc": "/asset/Skenario Keempat/Pengujian/kurvaroc.png"
        }
    ]

    for sk in skenario_list:
        st.header(sk["judul"])

        # --- Pelatihan ---
        st.subheader("Hasil Pelatihan (Fold Terbaik)")
        st.dataframe(pd.DataFrame(sk["train_table"]), use_container_width=True)

        col_acc, col_loss = st.columns(2)
        with col_acc:
            st.image(sk["train_imgs"][0], caption="Akurasi - Fold Terbaik", use_container_width=True)
        with col_loss:
            st.image(sk["train_imgs"][1], caption="Loss - Fold Terbaik", use_container_width=True)

        # --- Pengujian ---
        st.subheader("Hasil Pengujian Model Terbaik")
        st.markdown(f"**Akurasi Pengujian:** {sk['test_akurasi']}")

        st.markdown("#### Tabel Metrik Evaluasi:")
        st.dataframe(pd.DataFrame(sk["test_metrics"]).set_index(""), use_container_width=True)

        st.markdown("#### Visualisasi Confusion Matrix:")
        st.image(sk["conf_matrix"], use_container_width=True)

        st.markdown("#### Kurva ROC:")
        st.image(sk["roc"], use_container_width=True)

        st.markdown("---")

# ===================== PREDIKSI =====================
elif menu == "Prediksi":
    st.title("Prediksi Kanker Kulit dari Gambar")

    model_file = st.file_uploader("Unggah model (.keras)", type=["keras"])
    if model_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp.write(model_file.read())
            tmp_path = tmp.name

        model = load_model(tmp_path)
        model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        st.success("Model berhasil dimuat.")

        uploaded_image = st.file_uploader("Upload gambar kulit", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Gambar Asli", use_container_width=True)

            if st.button("Preprocessing"):
                resized = image.resize((64, 64))
                arr = np.expand_dims(np.array(resized) / 255.0, axis=0)
                st.session_state["img_array"] = arr
                st.session_state["img_resized"] = resized
                st.success("Preprocessing selesai.")

            if "img_array" in st.session_state:
                st.image(st.session_state["img_resized"], caption="Gambar Resize", use_container_width=True)
                if st.button("Prediksi"):
                    pred = model.predict(st.session_state["img_array"])
                    idx = np.argmax(pred)
                    label = label_encoder.inverse_transform([idx])[0]
                    st.subheader("Hasil Prediksi")
                    st.write(f"Label: **{label}**")
                    st.write(f"Probabilitas: {np.max(pred) * 100:.2f}%")
    else:
        st.info("Silakan unggah model terlebih dahulu.")

# ===================== TENTANG =====================
elif menu == "Tentang Penelitian":
    st.title("Tentang Penelitian")

    st.markdown("""
    ### Peneliti:
    - **Nama:** Soni Indra Maulana
    - **NIM:** 210411100136
    - **Institusi:** Universitas Trunojoyo Madura
    - **Pembimbing:** (Opsional)
      - Pembimbing 1: Prof. Arif Muntasa, S.Si., M.T.
      - Pembimbing 2: Prof. Rima Tri Wahyuningrum, S.T., M.T.


    ### Tujuan Penelitian
    Mengembangkan sistem klasifikasi citra kanker kulit menggunakan CNN berbasis MobileNetV2 serta membandingkan pendekatan manual vs transfer learning, dan mengatasi ketidakseimbangan data menggunakan teknik Random Oversampling (ROS).
    """)