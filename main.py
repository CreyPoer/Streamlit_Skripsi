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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import tambahan

# Sidebar Navigation
menu = st.sidebar.selectbox(
    "Pilih Halaman",
    ("Beranda", "Preprocessing", "Dataset HAM10000", "Pelatihan Model", "Evaluasi Model", "Prediksi", "Tentang Penelitian")
)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])

# ===================== BERANDA =====================
if menu == "Beranda":
    st.title("Implementasi Random Oversampling pada Klasifikasi Kanker Kulit pada Citra Dermoskopi Menggunakan Metode MobileNetV2")

    st.markdown("""
    ### Latar Belakang
    Kanker kulit merupakan salah satu jenis kanker yang umum terjadi di Asia, yang disebabkan oleh paparan sinar ultraviolet berlebih dari matahari sehingga mendorong pertumbuhan sel secara abnormal. Deteksi dini yang akurat sangat penting untuk mencegah penyebaran kanker lebih lanjut. Namun, proses diagnosis yang dilakukan oleh tenaga medis sering kali bersifat manual, sehingga berisiko menimbulkan kesalahan deteksi.

    Model klasifikasi berbasis komputer yang handal sangat dibutuhkan untuk membantu mengidentifikasi berbagai jenis kanker kulit secara otomatis. Salah satu metode yang terbukti efektif adalah **Convolutional Neural Network (CNN)** karena mampu mengenali pola-pola sel kanker kulit melalui citra dermoskopi, yang mengandung informasi penting seperti warna, tekstur, dan ukuran.

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
    - Pembagian dataset dilakukan menjadi **80% data latih dan 20% data uji**.
    - Untuk menyeimbangkan data latih, digunakan teknik **Random Oversampling (ROS)**.
    - Model klasifikasi yang digunakan adalah **MobileNetV2**.
    - Pendekatan **Transfer Learning ImageNet** dengan **fine-tuning (membuka semua layer base model)** diuji.
    - **Hyperparameter** yang digunakan adalah: learning rate = 0.001, batch size = 32, epoch = 50, optimizer = SGD (momentum = 0.9), callback ModelCheckpoint dan EarlyStopping, serta loss sparse_categorical_crossentropy.
    - Evaluasi model menggunakan **5-fold cross-validation** pada data latih untuk menguji generalisasi model selama pelatihan.
    """)

    st.markdown("### Perancangan Sistem")
    st.image("asset/perancangan-sistem.png", caption="Diagram Perancangan Sistem")

    st.markdown("### Skenario Uji Coba")

    skenario_df = pd.DataFrame({
        "No": [1, 2, 3, 4, 5, 6, 7, 8],
        "Random Oversampling": ["Pakai", "Pakai", "Tanpa", "Tanpa", "Pakai", "Pakai", "Tanpa", "Tanpa"],
        "Transfer learning": ["ImageNet (fine-tuning)", "ImageNet (fine-tuning)", "ImageNet (fine-tuning)", "ImageNet (fine-tuning)", "Tanpa", "Tanpa", "Tanpa", "Tanpa"],
        "Augmentasi dengan Image-Data-Generator": ["Pakai", "Tanpa", "Pakai", "Tanpa", "Pakai", "Tanpa", "Pakai", "Tanpa"]
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
    example_img_path = "asset/gambarpreprocessing.jpg"
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
        st.warning("Gambar contoh tidak ditemukan. Pastikan file `gambarpreprocessing.jpg` berada di direktori yang sama.")

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
            st.image(red_only.astype("uint8"), caption="Merah (dalam RGB)")
            st.caption(f"Ukuran: {resized_image[:, :, 0].shape}")  # (64, 64)
            st.code(f"{R[:10]} ...")

        with col2:
            st.markdown("**Channel Hijau (G):**")
            green_only = np.zeros_like(resized_image)
            green_only[:, :, 1] = resized_image[:, :, 1] * 255
            st.image(green_only.astype("uint8"), caption="Hijau (dalam RGB)")
            st.caption(f"Ukuran: {resized_image[:, :, 1].shape}")
            st.code(f"{G[:10]} ...")

        with col3:
            st.markdown("**Channel Biru (B):**")
            blue_only = np.zeros_like(resized_image)
            blue_only[:, :, 2] = resized_image[:, :, 2] * 255
            st.image(blue_only.astype("uint8"), caption="Biru (dalam RGB)")
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

    st.dataframe(label_df)

    # ========== Sub Preprocessing - Penerapan Random Oversampling pada Data Train ==============

    st.subheader("4. Tahap Balancing Data dengan Random Oversampling")

    st.markdown("""
    Pada tahap ini dilakukan proses **balancing dataset** menggunakan teknik **Random Oversampling (ROS)** pada data citra yang sebelumnya telah diubah menjadi bentuk numerik (1 dimensi) melalui proses transformasi.

    Tujuan dari ROS adalah untuk meningkatkan jumlah sampel pada **kelas minoritas** dengan cara menduplikasi secara acak data dari kelas tersebut, hingga jumlahnya setara dengan kelas mayoritas.

    Berikut adalah diagram alur proses ROS dalam penelitian ini:
    """)

    st.image("asset/alurros.png", caption="Alur Proses Random Oversampling")

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

    st.image("asset/hasilros.png", caption="Perbandingan Data Train Sebelum dan Sesudah ROS")

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
    except NameError: # Handle case where resized_image or transformed might not be defined if example_img_path doesn't exist
        st.warning("Gagal melakukan reshape. Pastikan data berbentuk 12288 (64x64x3) dan gambar contoh tersedia.")


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
    # Mengubah konfigurasi sesuai permintaan
    aug_config_new = {
        "Teknik Augmentasi": ["width_shift_range", "horizontal_flip", "vertical_flip"],
        "Nilai": ["0.05", "True", "True"]
    }
    st.table(pd.DataFrame(aug_config_new))

    st.markdown("#### Kode Implementasi:")
    st.code("""
# Membuta fungsi ImageDataGenerator untuk data train
train_datagen = ImageDataGenerator(
    rescale=1./255,              
    width_shift_range=0.05,
    horizontal_flip=True,    
    vertical_flip=True, 
)

# Membuta fungsi ImageDataGenerator untuk data validasi
val_datagen = ImageDataGenerator(rescale=1./255) # Validasi hanya dinormalisasi, tidak di-augmentasi
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

        # Mengubah parameter ImageDataGenerator sesuai permintaan untuk simulasi
        train_datagen_sim = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
        )

        st.markdown("### Simulasi Dampak Masing-Masing Teknik Augmentasi")

        col1, col2, col3 = st.columns(3) # Hanya 3 kolom karena rotasi dan zoom dihapus

        # Untuk simulasi, kita harus memastikan ada variasi jika nilai flip diatur ke True
        # Karena horizontal_flip dan vertical_flip adalah boolean, kita bisa coba generate beberapa kali
        # atau menunjukkan efeknya jika diaktifkan. Untuk visualisasi, kita hanya bisa menunjukkan satu hasil acak.
        
        # Simulasi width_shift_range
        with col1:
            st.markdown("**Geser Horizontal (0.05)**")
            shift_w_img = next(train_datagen_sim.flow(img_array_expanded, batch_size=1, seed=42))[0] # Pakai seed untuk konsistensi
            st.image(shift_w_img, caption="Geser X")

        # Simulasi horizontal_flip
        with col2:
            st.markdown("**Horizontal Flip**")
            # Untuk flip, kita bisa membalik gambar secara manual untuk menunjukkan efeknya
            flipped_h_img_array = np.flip(img_array_expanded, axis=2) # Flip horizontal
            flipped_h_img = train_datagen_sim.rescale * flipped_h_img_array[0] # Normalisasi
            st.image(flipped_h_img, caption="Flip Horizontal")

        # Simulasi vertical_flip
        with col3:
            st.markdown("**Vertical Flip**")
            # Untuk flip, kita bisa membalik gambar secara manual untuk menunjukkan efeknya
            flipped_v_img_array = np.flip(img_array_expanded, axis=1) # Flip vertical
            flipped_v_img = train_datagen_sim.rescale * flipped_v_img_array[0] # Normalisasi
            st.image(flipped_v_img, caption="Flip Vertikal")
            
        # NORMALISASI
        st.markdown("### Perbandingan Nilai Piksel Sebelum dan Sesudah Normalisasi")
        st.write("Berikut ini adalah nilai piksel pada channel R (merah) sebelum dan sesudah dilakukan normalisasi:")

        # Ambil piksel sebelum normalisasi
        st.markdown("**Sebelum Normalisasi (Rentang 0–255):**")
        st.code(img_array[:, :, 0][:5, :5])  # channel R gambar asli (tanpa rescale)

        # NORMALISASI - ambil contoh dari gambar yang sudah diproses oleh generator
        st.markdown("**Sesudah Normalisasi (Rentang 0–1):**")
        # Generate satu gambar lagi dengan generator_sim untuk mendapatkan normalisasi
        normalized_img_example = next(train_datagen_sim.flow(img_array_expanded, batch_size=1, seed=42))[0]
        st.code(normalized_img_example[:, :, 0][:5, :5])  # channel R gambar augmentasi (sudah dinormalisasi)


    else:
        st.warning("Gambar contoh tidak ditemukan. Pastikan `gambarpreprocessing.jpg` tersedia.")

# ===================== DATASET HAM10000 =====================
elif menu == "Dataset HAM10000":
    st.title("Dataset HAM10000: Skin Cancer MNIST")

    st.markdown("""
    Dataset yang digunakan dalam penelitian ini adalah **Skin Cancer MNIST: HAM10000**, yang dapat diakses melalui Kaggle: [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

    Dataset ini merupakan kumpulan citra dermoskopi yang sangat besar, berisi total **10.015 gambar** yang dikelompokkan ke dalam **tujuh jenis kanker kulit** yang berbeda. Citra-citra ini dikumpulkan selama 20 tahun dari dua lokasi berbeda: Departemen Dermatologi di Universitas Kedokteran Wina, Austria, dan praktik kanker kulit Cliff Rosendahl di Queensland, Australia. Setiap gambar dermoskopi dilengkapi dengan metadata yang diekstrak dari Powerpoint dan basis data Excel.

    **HAM10000** menyediakan variasi bentuk, tekstur, dan warna lesi kanker kulit dari berbagai penderita, menjadikannya dataset yang komprehensif untuk tugas klasifikasi citra kanker kulit.
    """)

    st.subheader("Sampel Gambar per Kelas")

    # Define the classes and their corresponding file names (assuming .jpeg and in 'asset/samples/' folder)
    class_info = {
        "akiec": "Actinic Keratoses and Intraepithelial Carcinoma / Bowen's Disease",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis-like Lesions",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevi",
        "vasc": "Vascular Lesions"
    }
    
    # Optional: Set a uniform width for sample images to ensure consistent display.
    # Adjust this value as needed based on your pre-processed image sizes.
    IMAGE_DISPLAY_WIDTH = 250 

    for label_key, class_name in class_info.items():
        image_path = f"asset/Sampel Dataset/{label_key}.jpg" # Adjust path and extension if needed
        if os.path.exists(image_path):
            st.markdown(f"**Kelas: {class_name} ({label_key})**")
            st.image(image_path, caption=f"Contoh citra {class_name}", width=IMAGE_DISPLAY_WIDTH)
            st.markdown("---")
        else:
            st.warning(f"Gambar sampel untuk kelas '{class_name}' tidak ditemukan di: {image_path}")

    st.subheader("Distribusi Data per Kelas")
    data_dist_df = pd.DataFrame({
        "No": [1, 2, 3, 4, 5, 6, 7],
        "Label": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "Nama Kelas": [
            "Actinic Keratoses and Intraepithelial Carcinoma / Bowen's Disease",
            "Basal Cell Carcinoma",
            "Benign Keratosis-like Lesions",
            "Dermatofibroma",
            "Melanoma",
            "Melanocytic Nevi",
            "Vascular Lesions"
        ],
        "Deskripsi": [
            "Prakanker kulit atau tahap awal kanker kulit akibat paparan sinar UV.",
            "Kanker kulit jenis non-melanoma yang paling umum.",
            "Jenis lesi kulit jinak yang sering diduga termasuk kedalam jenis kanker kulit karena serupa.",
            "Lesi kulit jinak yang sering ditemukan di kaki atau tangan.",
            "Jenis kanker yang sangat mudah menyebar dan sering ditemukan di kaki atau tangan.",
            "Lesi kulit jinak yang sering disebut Tahi Lalat.",
            "Termasuk kedalam lesi vaskular seperti angioma atau hemangioma."
        ],
        "Jumlah Data": [327, 514, 1099, 115, 1113, 6705, 142]
    })
    st.table(data_dist_df.set_index("No"))

    # New table for data split and validation data
    st.subheader("Pembagian Data Latih, Validasi, dan Uji")
    
    # Raw data for training and testing
    train_counts = [262, 411, 879, 92, 890, 5364, 114]
    test_counts = [65, 103, 220, 23, 223, 1341, 28]

    # Calculate "Data Latih" (80% of "Jumlah di Data Train")
    data_latih_counts = [round(0.80 * count) for count in train_counts]
    
    # Calculate "Data Validasi" (20% of "Jumlah di Data Train")
    data_validasi_counts = [round(0.20 * count) for count in train_counts]

    # Combine into a dictionary for DataFrame
    split_data = {
        "No": [1, 2, 3, 4, 5, 6, 7, "Total"],
        "Ragam Kelas": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", ""],
        "Data Train": train_counts + [sum(train_counts)],
        "Data Latih (80%)": data_latih_counts + [sum(data_latih_counts)],
        "Data Validasi (20%)": data_validasi_counts + [sum(data_validasi_counts)],
        "Data Test": test_counts + [sum(test_counts)],
    }

    split_df = pd.DataFrame(split_data)
    st.table(split_df.set_index("No"))

# ===================== PELATIHAN =====================
elif menu == "Pelatihan Model":
    st.title("Pelatihan Model Klasifikasi Citra Kanker Kulit")

    st.markdown("""
    Pada tahap ini dilakukan pelatihan model klasifikasi citra kanker kulit dengan dua pendekatan berbeda:

    1. **Model CNN Manual (MobileNetV2 from Scratch)**: Model MobileNetV2 dilatih dari awal tanpa menggunakan bobot pre-trained ImageNet.
    2. **Model Transfer Learning (MobileNetV2 Pre-trained)**: Menggunakan model MobileNetV2 yang telah dilatih pada dataset ImageNet, lalu dilakukan fine-tuning (semua layer base model dibuat *trainable*).

    Untuk menjamin generalisasi model, proses pelatihan dilakukan menggunakan **5-Fold Cross Validation**.
    """)

    # --- CNN Manual ---
    st.subheader("1. Model CNN Manual (MobileNetV2 from Scratch)")

    st.markdown("""
    Model ini dibangun dengan menyusun sendiri layer-layer konvolusi dan blok *inverted residual* seperti arsitektur MobileNetV2 asli, namun tanpa memanfaatkan bobot pre-trained. Model dilatih dari awal menggunakan data yang telah seimbang dan ditransformasikan.
    """)

    st.image("asset/mobilenetv2manual.jpg", caption="Struktur Arsitektur CNN Manual (MobileNetV2 from Scratch)")

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
    st.dataframe(df_manual)

    st.markdown("#### Kode Arsitektur CNN Manual:")
    st.code("""
# Fungsi dan implementasi blok residual sudah dijelaskan dalam model
# Berikut hanya ringkasan fungsi utama
model = create_model() # Diasumsikan ada fungsi create_model() untuk MobileNetV2 from scratch
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    """)

    # --- Transfer Learning ---
    st.subheader("2. Model Transfer Learning (MobileNetV2 Pre-trained)")

    st.markdown("""
    Model ini menggunakan arsitektur **MobileNetV2** yang telah dilatih sebelumnya pada dataset ImageNet. Teknik ini dikenal sebagai **Transfer Learning**. Semua layer dari model pre-trained dibuat **trainable** (fine-tuned) agar dapat menyesuaikan terhadap citra dermoskopi yang digunakan.
    """)

    st.image("asset/pretrainedmobilenetv2.jpg", caption="Struktur Transfer Learning MobileNetV2")

    st.markdown("#### Kode Implementasi:")
    st.code("""
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def create_tl_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    base_model.trainable = True # Fine-tuning: semua layer trainable

    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(7, activation="softmax")(x) # 7 kelas kanker kulit

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
    Untuk menjamin generalisasi model dan mengurangi risiko *overfitting*, proses pelatihan dilakukan menggunakan **5-Fold Cross Validation**. Data latih dibagi menjadi 5 bagian, model dilatih 5 kali dengan rotasi bagian data validasi, lalu hasilnya dirata-rata untuk menilai kinerja secara umum.

    Teknik ini meningkatkan keandalan evaluasi model.
    """)

    st.markdown("#### Ilustrasi:")
    st.image("asset/kfold_ilustrasi.PNG", caption="Skema 5-Fold Cross Validation")


# ===================== EVALUASI =====================
elif menu == "Evaluasi Model":
    st.title("Evaluasi Model Berdasarkan Skenario Uji Coba")

    st.markdown("""
    Evaluasi dilakukan terhadap **delapan skenario** pengujian untuk menilai performa model klasifikasi. Setiap skenario mencerminkan kombinasi penggunaan **Random Oversampling (ROS)**, **Transfer Learning**, dan **`ImageDataGenerator`**:
    """)

    # Data per skenario (8 skenario)
    skenario_data = [
        # Skenario 1: ROS + TL + IDG
        {
            "judul": "Skenario Pertama (ROS + Transfer Learning + ImageDataGenerator)",
            "waktu_pelatihan": "8174.74 detik (~2 jam 16 menit)",
            "train_avg_table": {
                "Metrik": ["Val Akurasi", "Val Loss"],
                "Rata-rata 5-Fold": ["99,14%", "3,87%"],
                "Fold Terbaik": ["99,37%", "2,95%"]
            },
            "train_imgs": [
                "asset/Skenario Pertama/Pelatihan/akurasi.png",
                "asset/Skenario Pertama/Pelatihan/loss.png"
            ],
            "test_summary": {
                "Acc": "81%",
                "Macro Avg P": "69%", "Macro Avg R": "68%", "Macro Avg F1": "69%",
                "Weighted Avg P": "82%", "Weighted Avg R": "81%", "Weighted Avg F1": "82%"
            },
            "test_class_metrics": {
                "Kelas": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "Presisi": ["59%", "73%", "60%", "52%", "57%", "92%", "92%"],
                "Recall": ["46%", "72%", "68%", "52%", "65%", "89%", "86%"],
                "F1-Score": ["52%", "72%", "64%", "52%", "60%", "91%", "89%"]
            },
            "test_auc_table": {
                "Kelas": ["Akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "AUC (%)": ["95", "98", "92", "98", "91", "94", "100"]
            },
            "conf_matrix": "asset/Skenario Pertama/Pengujian/confusionmatrix.png",
            "roc": "asset/Skenario Pertama/Pengujian/kurvaroc.png"
        },
        # Skenario 2: ROS + TL (Tanpa IDG)
        {
            "judul": "Skenario Kedua (ROS + Transfer Learning)",
            "waktu_pelatihan": "3116.94 detik (~52 menit)",
            "train_avg_table": {
                "Metrik": ["Val Akurasi", "Val Loss"],
                "Rata-rata 5-Fold": ["99,32%", "3,73%"],
                "Fold Terbaik": ["99,41%", "3,13%"]
            },
            "train_imgs": [
                "asset/Skenario Kedua/Pelatihan/akurasi.png",
                "asset/Skenario Kedua/Pelatihan/loss.png"
            ],
            "test_summary": {
                "Acc": "80%",
                "Macro Avg P": "67%", "Macro Avg R": "64%", "Macro Avg F1": "64%",
                "Weighted Avg P": "79%", "Weighted Avg R": "80%", "Weighted Avg F1": "79%"
            },
            "test_class_metrics": {
                "Kelas": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "Presisi": ["53%", "59%", "66%", "75%", "53%", "89%", "74%"],
                "Recall": ["58%", "67%", "60%", "39%", "46%", "91%", "82%"],
                "F1-Score": ["55%", "63%", "63%", "51%", "49%", "90%", "78%"]
            },
            "test_auc_table": {
                "Kelas": ["Akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "AUC (%)": ["96", "96", "89", "93", "88", "92", "100"]
            },
            "conf_matrix": "asset/Skenario Kedua/Pengujian/confusionmatrix.png",
            "roc": "asset/Skenario Kedua/Pengujian/kurvaroc.png"
        },
        # Skenario 3: Tanpa ROS + TL + IDG
        {
            "judul": "Skenario Ketiga (Tanpa ROS + Transfer Learning + ImageDataGenerator)",
            "waktu_pelatihan": "1584.63 detik (~26 menit)",
            "train_avg_table": {
                "Metrik": ["Val Akurasi", "Val Loss"],
                "Rata-rata 5-Fold": ["78,63%", "74,32%"],
                "Fold Terbaik": ["79,6%", "67,17%", ]
            },
            "train_imgs": [
                "asset/Skenario Ketiga/Pelatihan/akurasi.png",
                "asset/Skenario Ketiga/Pelatihan/loss.png"
            ],
            "test_summary": {
                "Acc": "79%",
                "Macro Avg P": "65%", "Macro Avg R": "55%", "Macro Avg F1": "58%",
                "Weighted Avg P": "78%", "Weighted Avg R": "79%", "Weighted Avg F1": "78%"
            },
            "test_class_metrics": {
                "Kelas": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "Presisi": ["32%", "63%", "66%", "50%", "56%", "87%", "100%"],
                "Recall": ["45%", "52%", "48%", "26%", "48%", "94%", "71%"],
                "F1-Score": ["37%", "57%", "56%", "34%", "52%", "90%", "83%"]
            },
            "test_auc_table": {
                "Kelas": ["Akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "AUC (%)": ["92", "97", "91", "96", "90", "92", "99"]
            },
            "conf_matrix": "asset/Skenario Ketiga/Pengujian/confusionmatrix.png",
            "roc": "asset/Skenario Ketiga/Pengujian/kurvaroc.png"
        },
        # Skenario 4: Tanpa ROS + TL (Tanpa IDG)
        {
            "judul": "Skenario Keempat (Tanpa ROS + Transfer Learning)",
            "waktu_pelatihan": "794.62 detik (~13 menit)",
            "train_avg_table": {
                "Metrik": ["Val Akurasi", "Val Loss"],
                "Rata-rata 5-Fold": ["76,59%", "124,2%"],
                "Fold Terbaik": ["78,28%", "116%"]
            },
            "train_imgs": [
                "asset/Skenario Keempat/Pelatihan/akurasi.png",
                "asset/Skenario Keempat/Pelatihan/loss.png"
            ],
            "test_summary": {
                "Acc": "77%",
                "Macro Avg P": "60%", "Macro Avg R": "51%", "Macro Avg F1": "54%",
                "Weighted Avg P": "75%", "Weighted Avg R": "77%", "Weighted Avg F1": "75%"
            },
            "test_class_metrics": {
                "Kelas": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "Presisi": ["41%", "68%", "53%", "23%", "60%", "85%", "90%"],
                "Recall": ["49%", "49%", "51%", "22%", "29%", "94%", "64%"],
                "F1-Score": ["44%", "57%", "52%", "22%", "39%", "89%", "75%"]
            },
            "test_auc_table": {
                "Kelas": ["Akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "AUC (%)": ["94", "94", "85", "89", "82", "89", "99"]
            },
            "conf_matrix": "asset/Skenario Keempat/Pengujian/confusionmatrix.png",
            "roc": "asset/Skenario Keempat/Pengujian/kurvaroc.png"
        },
        # Skenario 5: ROS + Tanpa TL + IDG
        {
            "judul": "Skenario Kelima (ROS + Tanpa Transfer Learning + ImageDataGenerator)",
            "waktu_pelatihan": "8943.86 detik (~2 jam 29 menit)",
            "train_avg_table": {
                "Metrik": ["Val Akurasi", "Val Loss"],
                "Rata-rata 5-Fold": ["96,32%", "10,9%"],
                "Fold Terbaik": ["96,87%", "9,56%"]
            },
            "train_imgs": [
                "asset/Skenario Kelima/Pelatihan/akurasi.png",
                "asset/Skenario Kelima/Pelatihan/loss.png"
            ],
            "test_summary": {
                "Acc": "72%",
                "Macro Avg P": "51%", "Macro Avg R": "56%", "Macro Avg F1": "53%",
                "Weighted Avg P": "75%", "Weighted Avg R": "72%", "Weighted Avg F1": "73%"
            },
            "test_class_metrics": {
                "Kelas": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "Presisi": ["33%", "65%", "49%", "21%", "37%", "90%", "66%"],
                "Recall": ["46%", "50%", "57%", "26%", "49%", "81%", "82%"],
                "F1-Score": ["38%", "56%", "53%", "23%", "42%", "85%", "73%"]
            },
            "test_auc_table": {
                "Kelas": ["Akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "AUC (%)": ["93", "91", "85", "90", "82", "91", "100"]
            },
            "conf_matrix": "asset/Skenario Kelima/Pengujian/confusionmatrix.png",
            "roc": "asset/Skenario Kelima/Pengujian/kurvaroc.png"
        },
        # Skenario 6: ROS + Tanpa TL (Tanpa IDG)
        {
            "judul": "Skenario Keenam (ROS + Tanpa Transfer Learning)",
            "waktu_pelatihan": "5165.80 detik (~1 jam 26 menit)",
            "train_avg_table": {
                "Metrik": ["Val Akurasi", "Val Loss"],
                "Rata-rata 5-Fold": ["98,89%", "5,35%"],
                "Fold Terbaik": ["99%", "4,98%"]
            },
            "train_imgs": [
                "asset/Skenario Keenam/Pelatihan/akurasi.png",
                "asset/Skenario Keenam/Pelatihan/loss.png"
            ],
            "test_summary": {
                "Acc": "73%",
                "Macro Avg P": "54%", "Macro Avg R": "49%", "Macro Avg F1": "51%",
                "Weighted Avg P": "73%", "Weighted Avg R": "73%", "Weighted Avg F1": "73%"
            },
            "test_class_metrics": {
                "Kelas": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "Presisi": ["44%", "48%", "43%", "33%", "46%", "86%", "78%"],
                "Recall": ["34%", "61%", "47%", "22%", "47%", "85%", "50%"],
                "F1-Score": ["38%", "54%", "45%", "26%", "47%", "86%", "61%"]
            },
            "test_auc_table": {
                "Kelas": ["Akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "AUC (%)": ["85", "87", "81", "87", "82", "86", "96"]
            },
            "conf_matrix": "asset/Skenario Keenam/Pengujian/confusionmatrix.png",
            "roc": "asset/Skenario Keenam/Pengujian/kurvaroc.png"
        },
        # Skenario 7: Tanpa ROS + Tanpa TL + IDG
        {
            "judul": "Skenario Ketujuh (Tanpa ROS + Tanpa Transfer Learning + ImageDataGenerator)",
            "waktu_pelatihan": "1667.49 detik (~27 menit)",
            "train_avg_table": {
                "Metrik": ["Val Akurasi", "Val Loss"],
                "Rata-rata 5-Fold": ["72,77%", "78,62%"],
                "Fold Terbaik": ["74,34%", "74,1%"]
            },
            "train_imgs": [
                "asset/Skenario Ketujuh/Pelatihan/akurasi.png",
                "asset/Skenario Ketujuh/Pelatihan/loss.png"
            ],
            "test_summary": {
                "Acc": "76%",
                "Macro Avg P": "50%", "Macro Avg R": "44%", "Macro Avg F1": "45%",
                "Weighted Avg P": "72%", "Weighted Avg R": "76%", "Weighted Avg F1": "73%"
            },
            "test_class_metrics": {
                "Kelas": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "Presisi": ["21%", "55%", "49%", "0%", "60%", "83%", "83%"],
                "Recall": ["5%", "51%", "52%", "0%", "31%", "94%", "71%"],
                "F1-Score": ["8%", "53%", "50%", "0%", "41%", "88%", "77%"]
            },
            "test_auc_table": {
                "Kelas": ["Akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "AUC (%)": ["91", "94", "87", "83", "85", "90", "97"]
            },
            "conf_matrix": "asset/Skenario Ketujuh/Pengujian/confusionmatrix.png",
            "roc": "asset/Skenario Ketujuh/Pengujian/kurvaroc.png"
        },
        # Skenario 8: Tanpa ROS + Tanpa TL (Tanpa IDG)
        {
            "judul": "Skenario Kedelapan (Tanpa ROS + Tanpa Transfer Learning)",
            "waktu_pelatihan": "738.93 detik (~12 menit)",
            "train_avg_table": {
                "Metrik": ["Val Akurasi", "Val Loss"],
                "Rata-rata 5-Fold": ["71,12%", "85,87%"],
                "Fold Terbaik": ["71,87%", "87,84%"]
            },
            "train_imgs": [
                "asset/Skenario Kedelapan/Pelatihan/akurasi.png",
                "asset/Skenario Kedelapan/Pelatihan/loss.png"
            ],
            "test_summary": {
                "Acc": "71%",
                "Macro Avg P": "37%", "Macro Avg R": "25%", "Macro Avg F1": "26%",
                "Weighted Avg P": "66%", "Weighted Avg R": "71%", "Weighted Avg F1": "65%"
            },
            "test_class_metrics": {
                "Kelas": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "Presisi": ["29%", "42%", "51%", "0%", "65%", "75%", "0%"],
                "Recall": ["8%", "32%", "32%", "0%", "6%", "98%", "0%"],
                "F1-Score": ["12%", "36%", "39%", "0%", "11%", "85%", "0%"]
            },
            "test_auc_table": {
                "Kelas": ["Akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "AUC (%)": ["88", "91", "83", "83", "82", "87", "88"]
            },
            "conf_matrix": "asset/Skenario Kedelapan/Pengujian/confusionmatrix.png",
            "roc": "asset/Skenario Kedelapan/Pengujian/kurvaroc.png"
        }
    ]

    for sk in skenario_data: # Menggunakan skenario_data untuk 8 skenario
        st.header(sk["judul"])

        # --- Pelatihan ---
        st.subheader("Hasil Pelatihan")
        st.markdown(f"**Waktu Total Pelatihan:** {sk['waktu_pelatihan']}") # Waktu pelatihan di luar tabel
        st.dataframe(pd.DataFrame(sk["train_avg_table"]).set_index("Metrik")) 

        col_acc, col_loss = st.columns(2)
        with col_acc:
            st.image(sk["train_imgs"][0], caption="Kurva Akurasi Pelatihan & Validasi")
        with col_loss:
            st.image(sk["train_imgs"][1], caption="Kurva Loss Pelatihan & Validasi")

        # --- Pengujian ---
        st.subheader("Hasil Pengujian Model Terbaik")
        st.markdown(f"**Akurasi Pengujian:** {sk['test_summary']['Acc']}")

        st.markdown("#### Tabel Metrik Evaluasi (Ringkasan):")
        st.dataframe(pd.DataFrame([sk["test_summary"]]))

        st.markdown("#### Tabel Metrik Evaluasi (Per Kelas):")
        st.dataframe(pd.DataFrame(sk["test_class_metrics"]).set_index("Kelas"))

        st.markdown("#### Tabel AUC (Area Under the Curve) Per Kelas:")
        st.dataframe(pd.DataFrame(sk["test_auc_table"]).set_index("Kelas"))

        st.markdown("#### Visualisasi Confusion Matrix:")
        st.image(sk["conf_matrix"])

        st.markdown("#### Kurva ROC:")
        st.image(sk["roc"])

        st.markdown("---")

# ===================== PREDIKSI =====================
# ===================== PREDIKSI =====================
elif menu == "Prediksi":
    st.title("Prediksi Kanker Kulit dari Gambar")

    # Path to your pre-loaded model
    # Pastikan Anda sudah menyimpan model terbaik (misal: dari Skenario 1)
    # ke dalam folder asset/models/ dengan nama 'best_mobilenetv2_model.keras'
    PRE_LOADED_MODEL_PATH = "best_model.keras" 

    model = None # Initialize model variable

    if os.path.exists(PRE_LOADED_MODEL_PATH):
        try:
            model = load_model(PRE_LOADED_MODEL_PATH)
            # Re-compile the model with the same optimizer and loss as during training
            # This is crucial if you saved only weights or if the optimizer state matters
            model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            st.success("Model klasifikasi telah berhasil dimuat secara otomatis.")
        except Exception as e:
            st.error(f"Gagal memuat model: {e}. Pastikan file model benar dan tidak rusak.")
            model = None
    else:
        st.warning("File model tidak ditemukan. Pastikan 'best_mobilenetv2_model.keras' ada di direktori 'asset/models/'.")


    if model: # Only proceed if model was loaded successfully
        uploaded_image = st.file_uploader("Unggah gambar kulit Anda untuk diprediksi", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Gambar Asli", use_container_width=True)

            # Preprocessing happens automatically before prediction
            # No need for a separate "Preprocessing" button for end-users
            
            st.subheader("Memproses dan Memprediksi...")
            
            # Perform preprocessing steps
            resized = image.resize((64, 64))
            # Ensure the image array is scaled correctly (0-1) as the model expects
            arr = np.expand_dims(np.array(resized) / 255.0, axis=0) 
            
            # Predict
            pred = model.predict(arr)
            idx = np.argmax(pred)
            
            # Inverse transform the predicted label
            # Ensure label_encoder is accessible here (it's global in your main.py)
            label = label_encoder.inverse_transform([idx])[0]
            
            st.subheader("Hasil Prediksi")
            st.write(f"Jenis Kanker Kulit yang Diprediksi: **{label.upper()}**") # Uppercase for clarity
            st.write(f"Keyakinan Model: **{np.max(pred) * 100:.2f}%**")
            st.info("Catatan: Prediksi ini hanya untuk tujuan informasi dan tidak menggantikan diagnosis medis profesional.")
    else:
        st.info("Model klasifikasi belum siap. Silakan periksa konfigurasi aplikasi.")

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