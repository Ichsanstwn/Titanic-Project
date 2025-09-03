import pandas as pd
import matplotlib.pyplot as plt

# 1) Memuat data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df_iris = pd.read_csv(url)

print("1) Kolom-kolom yang terdapat pada dataset Iris:")
print(df_iris.columns)

print("\n2) Inspeksi awal")
print("a) 5 baris pertama:")
print(df_iris.head(5))

print("\nb) 5 baris terakhir:")
print(df_iris.tail(5))

print("\nc) Info tentang dataset:")
print(df_iris.info())
print()

print(df_iris.describe())
print()

print(df_iris['species'].value_counts())

print("\n3) Visualisasi Data")
# Histogram: Membuat histogram untuk setiap fitur
df_iris.hist(figsize=(10, 8)) # .hist() -> fungsi yang disediakan oleh Pandas untuk membuat histogram (Pandas secara otomatis membuat histogram untuk setiap kolom numerik)
# figsize(10, 8) -> parameter yang mengatur ukuran gambar plot, (10,8) berarti lebar gambar 10 inch dan tinggi 8 inch

plt.suptitle("a) Histogram: Melihat Distribusi Setiap Fitur", fontsize=16) # .suptitle() -> fungsi dari Matplotlib untuk membuat judul utama
plt.show() # .show() -> fungsi yang memerintahkan Matplotlib untuk menampilkan semua plot yang telah dibuat (tanpa baris ini, plot mungkin tidak akan muncul di beberapa lingkungan)

# Box Plot: Membuat box plot untuk setiap fitur, dibagi berdasarkan spesies
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle("b) Box Plot: Mengidentifikasi Outliers dan Membandingkan Distribusi Antar Spesies")

# Petal Length
df_iris.boxplot(column='petal_length', by='species', ax=axes[0, 0])
# Petal Width
df_iris.boxplot(column='petal_width', by='species', ax=axes[0, 1])
# Sepal Length
df_iris.boxplot(column='sepal_length', by='species', ax=axes[1, 0])
# Sepal Width
df_iris.boxplot(column='sepal_width', by='species', ax=axes[1, 1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Scatter Plot: Memilih dua fitur yang paling membedakan -> petal length dan petal width
