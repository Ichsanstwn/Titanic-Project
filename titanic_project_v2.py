# Tujuan utama dari proyek ini adalah untuk membangun model Machine Learning yang dapat memprediksi apakah seorang penumpang akan selamat dari bencana Titanic atau tidak, berdasarkan data yang tersedia.

import pandas as pd
from sklearn.model_selection import train_test_split # untuk membagi dataset menjadi dua bagian: training set dan testing set
from sklearn.linear_model import LogisticRegression # mengimpor model Regresi Logistik
from sklearn.preprocessing import StandardScaler # digunakan untuk standarisasi data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # mengimpor beberapa metrik evaluasi yang umum digunakan
from sklearn.model_selection import GridSearchCV # mengimpor sebuah teknik yang akan digunakan untuk menemukan kombinasi hyperparameter yang paling optimal untuk meningkatkan kinerja model

# 1. Memuat Data (Data Loading)
# a) Memuat data dari file CSV, baik itu dari komputer kita atau dari sebuah URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df_titanic = pd.read_csv(url)

print("Proyek Titanic with Gemini\nKolom yang ada di df_titanic saat ini:")
print(df_titanic.columns)

# 2. Inspeksi Awal Data (Initial Data Inspection)
# a) Mencetak 5 baris pertama
print("\n1. Inspeksi Awal Data\na) Mencetak 5 baris pertama dari dataset")
print("="*40)
print(df_titanic.head(5))

# b) Mencetak informasi umum tentang DataFrame
print("\nb) Mencetak informasi umum tentang dataset")
print("="*40)
print(df_titanic.info())

# c) Menghitung total nilai NaN pada setiap kolom
print("\nc) Menghitung total nilai NaN pada setiap kolom")
print("="*40)
print(df_titanic.isna().sum())

# d) Menghasilkan statistik deskriptif dari kolom-kolom numerik dalam DataFrame (e.g., jumlah, rata-rata, standar deviasi, nilai minimum, kuartil dan nilai maksimum)
print("\nd) Menghasilkan statistik deskriptif dari kolom-kolom numerik dalam dataset")
print("="*40)
print(df_titanic.describe())

# 3. Eksplorasi Data Kategorikal (Categorical Data Exploration)
# value_counts() -> menghitung jumlah kemunculan (frekuensi) setiap nilai unik dalam sebuah kolom, untuk memahami distribusi data kategorikal (e.g., berapa banyak pria/wanita). parameter (opsional): normalize=True untuk melihat persentase/proporsi

# a) Menunjukkan berapa kali male dan female muncul, etc.
print("\n2. Eksplorasi Data Kategorikal\na) Menunjukkan berapa kali male dan female muncul, etc.")
print("="*40)
print(df_titanic['Sex'].value_counts())
print(df_titanic['Pclass'].value_counts(normalize=True)) # Pclass = Passenger class atau kelas penumpang -> 1: First class, 2: Second class, 3: Third class
# normalize=True -> mencetak proporsi atau persentas dari masing-masing nilai unik, bukan jumlah totalnya
print(df_titanic['Survived'].value_counts())

# 4. Eksplorasi Hubungan Antar Kolom (Relational Exploration/Aggregation)
# a) Berapa banyak penumpang yang selamat dan tidak selamat berdasarkan jenis kelamin
print("\n3. Eksplorasi Hubungan Antar Kolom\na) Berapa banyak penumpang yang selamat dan tidak selamat berdasarkan jenis kelamin")
print("="*40)
print(df_titanic.groupby('Sex')['Survived'].value_counts()) # hasilnya -> 0: Tidak selamat, 1: Selamat

# b) Hitung rata-rata usia dan tarif berdasarkan status selamat/tidak selamat -> Apakah ada perbedaan rata-rata usia atau rata-rata tarif antara penumpang yang selamat dan tidak selamat?
print("\nb) Hitung rata-rata usia dan tarif berdasarkan status selamat/tidak selamat")
print("="*40)
print(df_titanic.groupby('Survived')[['Age', 'Fare']].mean()) # fare -> tarif tiket

'''
(1) df_titanic.groupby('Survived) -> mengelompokkan semua baris di df_titanic berdasarkan nilai di kolom 'Survived' (0 atau 1). [['Age', 'Fare']] -> setelah dikelompokkan, kita hanya tertarik pada kolom 'Age' dan 'Fare'
(2) .mean() -> menghitung rata-rata dari 'Age' dan 'Fare' untuk setiap kelompok 'Survived' (rata-rata usia/tarif untuk yang tidak selamat, dan rata-rata usia/tarif untuk yang selamat)
'''

# 5. Persiapan Data (Data Preprocessing)
# a) Menerapkan one-hot encoding pada kolom 'Sex'
df_sex_encoded = pd.get_dummies(df_titanic['Sex'], prefix='Sex')

'''
(1) pd.get_dummies() -> fungsi utama Pandas untuk melakukan proses One-Hot Encoding. df_titanic['Sex'] -> input untuk fungsi get_dummies(). Kita memberikan kolom 'Sex' dari DataFrame df_titanic. prefix='Sex' -> parameter opsional yang sangat berguna. Ketika get_dummies() membuat kolom-kolom baru, ia akan memberi nama kolom-kolom itu dengan menambahkan awalan ini.
Jadi, alih-alih hanya 'female' dan 'male', kolomnya akan menjadi 'Sex_female' dan 'Sex_male'. Ini membantu kita mengidentifikasi bahwa kolom-kolom baru ini berasal dari kolom 'Sex' yang asli. df_sex_encoded -> Hasil dari proses encoding akan disimpan dalam variabel baru ini
'''

# b) Cetak 5 baris pertama dari DataFrame hasil encoding
print("\n4. Menerapkan One-Hot Encoding pada Kolom 'Sex'\na) Mencetak 5 baris pertama dari kolom 'Sex' hasil One-Hot Encoding")
print("="*40)
print(df_sex_encoded.head())

# c) Kita juga bisa melihat tipe datanya
print("\nb) Melihat informasi tipe data hasil encoding kolom 'Sex'")
print("="*40)
print(df_sex_encoded.info())

# d) Menggabungkan df_sex_encoded ke df_titanic
# pd.concat() menggabungkan DataFrame. axis=1 berarti menggabungkan secara kolom (horizontal)
df_titanic = pd.concat([df_titanic, df_sex_encoded], axis=1)

# e) Menghapus kolom 'Sex' yang asli
df_titanic.drop(columns=['Sex'], inplace=True)

"""
(1) .drop() menghapus kolom atau baris. columns=['Sex'] untuk spesifik kolom.
(2) inplace=True berarti perubahan langsung diterapkan pada df_titanic
"""

# f) Menampilkan beberapa baris pertama dari DataFrame yang sudah diubah
print("\nc) df_titanic setelah One-Hot Encoding 'Sex' dan penghapusan kolom asli")
print("="*40)
print(df_titanic.head())

# g) Menampilkan informasi DataFrame untuk konfirmasi tipe data dan kolom
print("\nd) Informasi df_titanic setelah perubahan")
print("="*40)
print(df_titanic.info())

# h) Mengecek nilai unik dan nilai hilang di kolom 'Embarked' (Embarked: Pelabuhan keberangkatan penumpang)
print("\n4.1. Mengecek nilai unik dan nilai hilang di kolom 'Embarked'")
print("="*40)
print(df_titanic['Embarked'].value_counts(dropna=False))

# 'S': Southampton, 'C': Cherbourg, 'Q': Queenstown

# i) Menangani nilai yang hilang pada kolom 'Embarked'
print("\nh) Menangani nilai yang hilang pada kolom 'Embarked'")
print("="*40)

'''
Ada dua pilihan, yaitu:
dropna() -> baris-baris yang memiliki nilai NaN akan dihapus seluruhnya. Karena hanya ada 2 NaN dari total 891 baris, menghapus 2 baris tidak akan terlalu memengaruhi ukuran dataset secara signifikan
fillna() -> memungkinkan kita untuk mengisi nilai NaN dengan nilai tertentu. Untuk kolom kategorikal, strategi yang paling umum adalah mengisi nilai NaN dengan modus (nilai yang paling sering muncul di kolom tersebut)
'''

# Menemukan modus dari kolom 'Embarked'
print("\nmenggunakan fillna() dengan cara menemukan modus dari kolom 'Embarked'")
embarked_mode = df_titanic['Embarked'].mode()[0]
print(f"Modus dari kolom 'Embarked' adalah: {embarked_mode}")

'''
Metode .mode() -> menghitung nilai-nilai yang paling sering muncul. Hasilnya adalah sebuah Pandas Series karena bisa saja ada lebih dari satu modus.
[0] -> Karena bisa saja lebih dari satu modus, kita mengambil elemen pertama. Fyi, Pandas Series: satu kolom tunggal atau array satu dimensi berlabel
'''

# j) Mengisi nilai yang hilang dengan modus (mode)
# df_titanic['Embarked'].fillna(embarked_mode, inplace=True) -> bisa pakai kode ini atau pake kode yang direkomendasikan untuk menghindari FutureWarning atau peringatan bahwa ada cara penulisan kode yang mungkin tidak berfungsi di versi Pandas mendatang
df_titanic['Embarked'] = df_titanic['Embarked'].fillna(embarked_mode)

# Memastikan bahwa tidak ada lagi nilai NaN di kolom 'Embarked'
print("\ni) Mengisi nilai yang hilang dengan modus (mode)\nMemastikan tidak ada lagi nilai NaN di kolom 'Embarked'")
print("="*40)
print(df_titanic['Embarked'].value_counts(dropna=False))

# k) Menerapkan one-hot encoding pada kolom 'Embarked'
df_embarked_encoded = pd.get_dummies(df_titanic['Embarked'], prefix='Embarked')

# l) Cetak 5 baris pertama dari DataFrame hasil encoding
print("\nl) Menerapkan One-Hot Encoding pada Kolom 'Embarked'\nl.1. Mencetak 5 baris pertama dari kolom 'Embarked' hasil One-Hot Encoding")
print("="*40)
print(df_embarked_encoded.head())

# m) Kita juga bisa melihat tipe datanya
print("\nl.2. Melihat informasi tipe data hasil encoding kolom 'Embarked'")
print("="*40)
print(df_embarked_encoded.info())

# n) Menggabungkan df_embarked_encoded ke df_titanic
# pd.concat() menggabungkan DataFrame. axis=1 berarti menggabungkan secara kolom (horizontal)
df_titanic = pd.concat([df_titanic, df_embarked_encoded], axis=1)

# o) Menghapus kolom 'Embarked' yang asli
df_titanic.drop(columns=['Embarked'], inplace=True)

"""
(1) .drop() menghapus kolom atau baris. columns=['Embarked'] untuk spesifik kolom.
(2) inplace=True berarti perubahan langsung diterapkan pada df_titanic
"""

# p) Menampilkan beberapa baris pertama dari DataFrame yang sudah diubah
print("\nm) df_titanic setelah One-Hot Encoding 'Embarked' dan penghapusan kolom asli")
print("="*40)
print(df_titanic.head())

# q) Menampilkan informasi DataFrame untuk konfirmasi tipe data dan kolom
print("\nn) Informasi df_titanic setelah perubahan")
print("="*40)
print(df_titanic.info())

# r) Periksa kembali total nilai yang hilang di seluruh DataFrame, untuk memastikan tidak ada lagi NaN yang terlewat
print("\nTotal nilai NaN per kolom setelah preprocessing")
print("="*40)
print(df_titanic.isna().sum())

# s) Menangani nilai NaN yang masih ditemukan
"""
Masih banyak ditemukan nilai NaN pada kolom 'Age' dan 'Cabin'. Maka dari itu, kita harus menangani baris-baris dengan nilai NaN pada masing-masing kolom. Karena total nilai NaN pada kedua kolom sangat banyak, kita tidak bisa menghapusnya secara langsung. 
Kita bisa menggunakan cara kedua, yaitu mengisi nilai NaN tersebut. Kolom 'Age' berisi data numerik, maka kita bisa mengisi nilai NaN itu dengan mencari nilai Median pada kolom 'Age'
"""

print("\nMenangani nilai NaN yang masih ditemukan")
print("="*40)
age_median = df_titanic['Age'].median()
print(f"\nMencari nilai median dari kolom 'Age'\nMedian dari kolom 'Age' adalah: {age_median}")

df_titanic['Age'] = df_titanic['Age'].fillna(age_median)
print("\nTotal nilai NaN per kolom setelah kita mengisi nilai NaN pada kolom 'Age' dengan mediannya")
print(df_titanic.isna().sum())

# t) Menghapus kolom 'Cabin'
"""
Setelah berhasil menangani nilai NaN pada kolom 'Age', selanjutnya kita akan menangani kolom 'Cabin'. Seperti yang sudah kita tampilkan di atas, kolom 'Cabin' memiliki 687 nilai NaN dari total 891 baris, yang artinya sekitar 77% datanya hilang
Karena persentase ini sangat tinggi, maka strategi terbaik untuk saat ini adalah menghapus seluruh kolom 'Cabin'
"""

df_titanic.drop(columns=['Cabin'], inplace=True)
print("\nTotal nilai NaN per kolom, setelah kolom 'Cabin' dihapus")
print(df_titanic.isna().sum())

print("\nInformasi df_titanic setelah semua preprocessing yang dilakukan")
print("="*40)
print(df_titanic.info())

# u) Menghapus kolom 'PassengerId', 'Name', dan 'Ticket'
"""
Berdasarkan info yang telah ditampilkan, masih terdapat beberapa kolom yang bertipe 'object' (yang berarti kolom tersebut berisi teks atau tipe data campurang yang tidak langsung numerik), yaitu 'PassengerId', 'Name', 'Ticket'
'PassengerId' hanyalah ID unik dan tidak memiliki kekuatan prediktif terhadap apakah seseorang selamat atau tidak. 'Name' dan 'Ticket' adalah kolom dengan kardinalitas yang sangat tinggi (banyak nilai unik). Mengubahnya langsung ke numerik (e.g., One-Hot Encoding)
akan menghasilkan terlalu banyak kolom baru yang sebagian besar berisi nol dan dapat membebani model dan tidak memberikan informasi yang berarti. Maka dari itu, pendekatan yang paling umum adalah menghapus kolom-kolom tersebut karena mereka tidak memberikan kontribusi langsung sebagai fitur prediktif
"""

# Menghapus beberapa kolom sekaligus
df_titanic.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

print("\nInformasi df_titanic setelah menghapus kolom-kolom yang tidak relevan")
print("="*40)
print(df_titanic.info())

print("\n5 baris pertama df_titanic setelah perubahan yang terjadi")
print("="*40)
print(df_titanic.head())

# Feature Engineering
# Mencipatkan fitur (variabel) baru dari fitur yang sudah ada di dalam dataset, untuk memberikan lebih banyak informasi yang relevan kepada model ML
# (i) Membuat fitur FamilySize
df_titanic['FamilySize'] = df_titanic['SibSp'] + df_titanic['Parch'] + 1
print("\nFitur FamilySize berhasil dibuat")

"""
(1) df_titanic['FamilySize'] -> Kita sedang membuat kolom baru di dalam DataFrame df_titanic dengan nama 'FamilySize'
(2) df_titanic['SibSp'] -> Kita mengambil nilai dari kolom jumlah saudara atau pasangan untuk setiap baris dan
(3) df_titanic['Parch'] -> Kita mengambil nilai dari kolom jumlah orang tua atau anak untuk setiap baris, kemudian dijumlahkan
(4) + 1 -> Kita menambahkan '1' untuk mewakili penumpang itu sendiri
Kode ini secara efektif menghitung total jumlah anggota keluarga yang bepergian bersama seorang penumpang dan menyimpannya di kolom baru
"""

# Membuat fitur IsAlone (Apakah penumpang bepergian sendirian)
df_titanic['IsAlone'] = df_titanic['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
print("\nFitur IsAlone berhasil dibuat")

"""
(1) df_titanic['IsAlone'] -> Kita membuat kolom baru bernama 'IsAlone'. Kolom ini akan menjadi fitur biner (berisi nilai 0 atau 1)
(2) df_titanic['FamilySize'].apply -> apply() menjalankan sebuah fungsi (dalam kasus ini, sebuah lambda function) untuk setiap nilai di dalam kolom 'FamilySize'
(3) lambda x: 1 if x == 1 else 0 -> Fungsi anonim, fungsi ini melakukan pengecekan: Jika nilai x (yaitu nilai di kolom FamilySize untuk setiap baris) = 1, maka fungsi mengembalikan nilai 1 (artinya 'bepergian sendirian')
jika tidak, fungsi akan mengembalikan nilai 0 (artinya 'bepergian dengan keluarga')
"""

# Data Preprocessing -> Persiapan Pelatihan Model: Langkah Awal dari Fase Pembagian Data (Data Splitting)
# Tugas pertama di fase ini: Memisahkan fitur (X) dari target (y)
"""
Fitur (X): Semua kolom yang akan kita gunakan sebagai input untuk model. Model akan belajar dari pola-pola di kolom-kolom ini.
Target (y): Kolom yang ingin kita prediksi. Dalam kasus kita, kolom target adalah kolom 'Survived' (Apakah penumpang selamat atau tidak?)
"""

# v) Memisahkan fitur (X) dan target (y)
# Tujuan: Mendefinisikan secara jelas apa yang akan menjadi masukan (input) dan apa yang akan menjadi keluaran (output) bagi model
X = df_titanic.drop('Survived', axis=1) # X adalah semua kolom (fitur) kecuali kolom 'Survived' -> axis=1 untuk kolom, axis=0 untuk baris/indeks
y = df_titanic['Survived'] # y adalah kolom 'Survived' (target)

print("\n5 baris pertama X (fitur)")
print("="*40)
print(X.head())

print("\n5 baris pertama y (target)")
print("="*40)
print(y.head())

# Membagi Data menjadi Set Pelatihan dan Set Pengujian (Data Splitting)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
(1) X adalah DataFrame yang berisi fitur-fitur, 
(2) y adalah Series yang berisi target (Series: Array NumPy 1-dimensi (satu urutan nilai/data)). test_size=0.2 -> parameter untuk menentukan proporsi data yang akan dialokasikan untuk testing set, 0.2 = 20% dari total data -> testing set, 0.8 = 80% -> training set (kita bisa mengubah nilai ini sesuai kebutuhan)
(3) random_state-42 -> parameter ini digunakan untuk menjaga konsistensi acakan, memastikan bahwa setiap kali Anda menjalankan kode yang sama dengan random_state yang sama, Anda akan mendapatkan hasil acakan yang persis sama. Memungkinkan reproduktivitas -> memungkinkan orang lain untuk mereplikasi hasil eksperimen ML Anda secara persis.
angka 42 -> pilihan yang populer dan konvensional, tetapi tidak memiliki kekuatan khusus apapun
"""

print("\nBentuk dari X_train:", X_train.shape) # fitur untuk melatih model
print("Bentuk dari X_test:", X_test.shape) # fitur untuk menguji model
print("Bentuk dari y_train:", y_train.shape) # target untuk melatih model
print("Bentuk dari y_test:", y_test.shape) # target untuk menguji model

"""
Atribut .shape akan mengembalikan sebuah tuple yang merepresentasikan dimensi dari objek tersebut. Untuk DataFrame akan mengembalikan tuple dalam format: (jumlah_baris, jumlah_kolom). Untuk Series yang merupakan 1 dimensi, .shape akan mengembalikan tuple dalam format: (jumlah_elemen)
e.g., hasil outputnya -> X_train: (712, 10) -> 712 baris dan 10 kolom
"""

# w) Feature Scaling -> Standarisasi Data
# Solusi terbaik untuk mengatasi masalah 'ConvergenceWarning'

# Inisialisasi StandardScaler
scaler = StandardScaler()

# Melakukan penskalaan pada X_train (fit dan transform)
# scaler.fit(X_train) -> mempelajari rata-rata dan standar deviasi dari X_train
# scaler.transform(X_train) -> menerapkan penskalaan tersebut
X_train_scaled = scaler.fit_transform(X_train)

# Melakukan penskalaan pada X_test (hanya transform, menggunakan rata-rata dan standar deviasi dari X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFitur-fitur numerik telah berhasil diskalakan")
print("="*40)
print("Bentuk X_train_scaled:", X_train_scaled.shape)
print("Bentuk X_test_scaled:", X_test_scaled.shape)

# 6. Melatih Model Machine Learning
# Langkah-langkah: (i) Memilih algortima model, (ii) Menginisialisasi model, (iii) Melatih model
# (i) Memilih algoritma model

"""
Algoritma yang kita pilih: Regresi Logistik.
Alasan: 
1) Masalah Titanic adalah masalah klasifikasi biner, kita ingin memprediksi salah satu dari dua kelas: 'Survived' (1) atau 'Not Survived' (0). Regresi Logistik dirancang khusus untuk memodelkan probabilitas suatu peristiwa biner.
2) Sederhana dan mudah diinterpretasi, 
3) Baseline model yang kuat -> Dalam banyak proyek Machine Learning, Regresi Logistik sering digunakan sebagai model dasar. Jika model sederhana ini sudah memberikan hasil yang cukup baik, kita bisa mempertimbangkan apakah perlu menggunakan model yang lebih kompleks atau tidak.
4) Cepat dan efisien,
5) Tidak sensitif terhadap skala fitur (pada kasus ini)
"""

# (ii) Menginisialisasi model

model_logistic_regression = LogisticRegression(random_state=42, max_iter=1000)

# max_iter=200 -> parameter ini mengatur jumlah iterasi maksimum yang diizinkan untuk algoritma mencoba menemukan titik konvergensi ini. Konvergensi algoritma mengacu pada titik dimana algoritma Machine Learning telah menemukan solusi "terbaik" atau "optimal" dan tidak akan ada lagi perbaikan signifkan yang dapat dicapai dengan melanjutkan prosesnya.

print("\nModel Regresi Logistik berhasil diinisialisasi")

# (iii) Melatih model

model_logistic_regression.fit(X_train_scaled, y_train) # fit() -> metode standar dalam library Scikit-learn (dan banyak library ML lainnya) yang digunakan untuk melatih model

"""
Jika muncul sebuah peringatan: ConvergenceWarning -> Ini bukan error yang menghentikan program, tetapi sebuah peringatan bahwa model yang dilatih mungkin belum mencapai kinerja terbaiknya atau belum sepenuhnya 'belajar' dari data.
Solusi: 
1) Meningkatkan max_iter (e.g., max_iter=1000 atau bahkan 5000) 
2) Melakukan Skala Data (Feature Scaling) -> Ini adalah solusi yang lebih fundamental dan seringkali sangat penting untuk algoritam berbasis gradien seperti Regresi Logistik
Mengapa diperlukan? Jika fitur-fitur kita memiliki rentang nilai yang sangat berbeda (e.g., 'Age' dari 0-80, 'Fare' dari 0-512), algoritma optimasi akan kesulitan. Fitur dengan nilai yang lebih besar bisa mendominasi perhitungan gradien, memperlambat atau bahkan mencegah konvergensi.
"""

print("\nModel Regresi Logistik berhasil dilatih!")

# Membuat prediksi

y_pred = model_logistic_regression.predict(X_test_scaled) # predict() -> metode standar pada objek model yang sudah dilatih di Scikit-learn (dan library ML lainnya). Fungsinya adalah untuk menghasilkan label prediksi berdasarkan data masukan baru
# perintah untuk meminta model yang sudah dilatih agar memprediksi status 'Survived' untuk setiap penumpang di set pengujian (X_test_scaled).

"""
Perbedaan fit() dan predict():
fit(): Ini adalah fase pembelajaran. Model menyesuaikan parameternya untuk memahami pola dalam data pelatihan.
predict(): Ini adalah fase aplikasi. Model menggunakan parameter yang sudah dipelajari untuk menghasilkan output (prediksi) pada data baru.
"""

print("\nPrediksi berhasil dibuat. Berikut adalah hasil 5 prediksi pertama:")
print("="*40)
print(y_pred[:5]) # -> menunjukkan bahwa output dari model adalah sebuah NumPy array

# 7. Evaluasi Model
# Kita perlu membandingkan y_pred (prediksi model) dengan y_test (nilai sebenarnya) untuk mengetahui seberapa akurat model kita

# x) Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAkurasi model: {accuracy:.4f}") # Akurasi adalah metrik paling sederhana, yaitu proporsi dari total prediksi yang dibuat model dengan benar, .4f adalah format untuk menampilkan akurasi sebagai angka desimal dengan 4 angka di belakang koma (e.g., 0.8547)
print(f"Classfication report: {classification_report(y_test, y_pred)}") # memberikan metrik evaluasi yang lebih mendalam untuk setiap kelas

"""
classification_report():
1) precision (presisi) -> Presisi yang tinggi berarti model tidak sering membuat kesalahan prediksi positif (False Positive)
2) recall (sensitivitas) -> Recall yang tinggi berarti model tidak sering melewatkan kasus positif yang sebenarnya (False Negative)
3) f-1 score -> rata-rata harmonik dari presisi dan recall
4) support -> jumlah sampel sebenarnya di set pengujian untuk setiap kelas
"""

print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}") # tabel yang menunjukkan jumlah prediksi benar dan salah yang dibuat oleh model, dipecah berdasarkan kelas

"""
Kesimpulan: Secara keseluruhan, model ini memiliki akurasi yang solid, yaitu sekitar 80,45%. Namun, laporan ini juga menunjukkan bahwa model kita sedikit lebih baik dalam memprediksi siapa yang tidak akan selamat (recall 86%)
dibandingkan dengan siapa yang akan selamat (recall 73%). Ini adalah hasil yang sangat bagus untuk model pertama kita.
"""

# 8. Meningkatkan kinerja model
# y) Hyperparameter turning
"""
Untuk menemukan kombinasi hyperparameter terbaik, kita tidak bisa menebak-nebak. Kita akan menggunakan teknik yang disebut 'Grid Search with Cross-Validation', yang akan secara sistematis
mencoba semua kombinasi hyperparameter yang kita tentukan dan menemukan yang paling optimal. Untuk Regresi Logistik, hyperparameter yang bisa kita ubah antara lain:
1) C -> Kekuatan regularisasi (regularization strength). Regularisasi adalah teknik untuk mencegah overfitting. Nilai 'C' yang lebih kecil berarti regularisasi yang lebih kuat
2) solver -> Algoritma yang digunakan untuk optimasi. Kita menggunakan 'lbfgs'. Ada pilihan lain seperti 'liblinear' atau 'saga'
"""

# Definisikan hyperparameter yang ingin kita uji
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'lbfgs', 'saga']
}

# Inisialisasi GridSearchCV
grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=5000, random_state=42), # max_iter lebih tinggi agar konvergen
    param_grid=param_grid,
    cv=5,
    scoring='accuracy', # Metrik yang digunakan untuk evaluasi
    n_jobs=-1 # Menggunakan semua core CPU untuk mempercepat
)

"""
(1) estimator: model yang ingin kita tuning
(2) param_grid: hyperparameter yang akan dicoba
(3) cv: jumlah lipatan (fold) untuk cross-validation
"""

# Jalankan grid search pada training data yang sudah diskalakan
grid_search.fit(X_train_scaled, y_train)

print("\nGrid Search selesai dan berhasil menemukan hyperparameter terbaik")
print("="*40)
print("Parameter terbaik:", grid_search.best_params_)
print("Skor terbaik:", grid_search.best_score_)

"""
(1) 'C': 0.1 -> Ini adalah nilai optimal untuk parameter regularisasi
(2) 'solver':'lbfgs' -> Algoritma optimasi 'lbfgs' yang kita gunakan dari awal terbukti menjadi pilihan terbaik dari opsi yang kita berikan
Skor terbaik: 0.7962966610853935 -> Angka ini adalah akurasi rata-rata yang dicapai oleh model dengan parameter terbaik selama cross-validation pada training set
Meskipun nilai ini sedikit lebih rendah dari akurasi awal kita di testing set (0.8045), itu wajar. Skor terbaik ini adalah nilai rata-rata, dan ini menandakan bahwa model yang kita temukan ini cenderung lebih stabil dan lebih baik dalam menghadapi data yang bervariasi
"""

# 9. Membuat Model Baru yang Final
# z) Menggunakan hyperparameter terbaik yang baru saja ditemukan
# Buat instance model baru dengan parameter terbaik

best_logistic_regression = LogisticRegression(C=0.1, solver='lbfgs', max_iter=5000, random_state=42)

# Latih model final ini pada seluruh training set yang sudah diskalakan
best_logistic_regression.fit(X_train_scaled, y_train)

print("\nModel final dengan hyperparameter terbaik telah dilatih")

# Membuat prediksi menggunakan model final pada testing set
y_pred_tuned = best_logistic_regression.predict(X_test_scaled)

# Menghitung akurasi model yang sudah di-tuning
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"\nAkurasi model (tuned): {accuracy_tuned:.4f}")
print(f"Classification report (tuned): {classification_report(y_test, y_pred_tuned)}")
print(f"Confusion matrix (tuned): {confusion_matrix(y_test, y_pred_tuned)}")

"""
Penurunan akurasi yang kecil ini menunjukkan bahwa model Regresi Logistik kita, setelah penskalaan data, sudah beroperasi mendekati potensi maksimalnya. Tuning hyperparameter tidak menghasilkan lompatan kinerja yang besar karena modelnya sudah cukup optimal.
Ini adalah pelajaran penting dalam Machine Learning: tidak semua tuning akan memberikan peningkatan dramatis, tetapi prosesnya membantu kita memastikan bahwa model kita adalah yang terbaik yang bisa kita dapatkan dari algoritma tertentu.
"""
