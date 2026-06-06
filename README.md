# Waste Classification AI Using Random Forest

## Deskripsi

Waste Classification AI adalah aplikasi berbasis web yang digunakan untuk mengklasifikasikan jenis sampah ke dalam tiga kategori:

* Organik
* Anorganik
* B3 (Bahan Berbahaya dan Beracun)

Aplikasi ini menggunakan algoritma Random Forest sebagai model machine learning dan Flask sebagai framework backend.

---

## Dataset

Dataset yang digunakan terdiri dari:

| Kategori  | Jumlah Data |
| --------- | ----------- |
| Organik   | 1037        |
| Anorganik | 1953        |
| B3        | 2636        |

Total dataset: 5626 gambar

---

## Teknologi yang Digunakan

* Python
* Flask
* Scikit-Learn
* OpenCV
* Bootstrap 5
* Random Forest Classifier

---

## Fitur

* Upload gambar sampah
* Prediksi kategori sampah
* Menampilkan confidence score
* Antarmuka berbasis Bootstrap
* Model Random Forest

---

## Cara Menjalankan

### Install Dependency

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_model.py
```

### Jalankan Aplikasi

```bash
python app.py
```

### Buka Browser

```text
http://127.0.0.1:5000
```

---

## Hasil Model

Akurasi Model:

68.03%

---

## Pengembang

Muhammad Faris
301240055
Teknik Informatika / 4C
Universitas Bale Bandung
"# Waste-Classification-AI" 
