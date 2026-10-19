# Klasifikasi Kualitas Kopi Menggunakan Machine Learning

## Deskripsi

Proyek ini menggunakan supervised learning untuk mengklasifikasikan kualitas
biji kopi dari karakteristik fisik dan informasi budidaya/pengolahan di dataset
Coffee Quality Ratings. Model yang dibandingkan adalah Decision Tree dan
Random Forest; Dummy Classifier digunakan sebagai baseline.

Model memprediksi kelas berdasarkan data tabular, bukan dari foto biji kopi.

## Tujuan

1. Memahami dan memeriksa dataset kualitas kopi.
2. Melakukan preprocessing pada fitur numerik dan kategorikal.
3. Membentuk kategori kualitas dari `total_cup_points`.
4. Membandingkan Decision Tree dan Random Forest dengan baseline.
5. Mengevaluasi Accuracy, Precision, Recall, F1-score, dan confusion matrix.

## Dataset dan target

Dataset: [Coffee Quality Ratings](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-07-07/coffee_ratings.csv).

Program menggunakan `coffee_ratings.csv` yang tersedia di folder proyek.
Jika file tersebut tidak ada, program akan mengunduhnya otomatis dari sumber
data publik; koneksi internet diperlukan untuk pengunduhan itu.

Kategori target adalah aturan eksperimen, bukan label resmi dalam dataset:

- **Kualitas Tinggi**: `total_cup_points >= 80`
- **Di Bawah Ambang**: `total_cup_points < 80`

Nilai `total_cup_points` hanya digunakan untuk membentuk target. Fitur
penilaian sensoris seperti aroma, flavor, aftertaste, acidity, body, balance,
uniformity, clean cup, sweetness, dan cupper points tidak digunakan sebagai
input agar tidak terjadi kebocoran target dari unsur penilaian yang
membentuk skor kualitas.

Fitur input yang digunakan:

- Numerik: kadar air, jumlah cacat, quakers, dan rentang/rerata ketinggian.
- Kategorikal: spesies, negara asal, wilayah, varietas, metode pemrosesan,
  dan warna.

Nilai yang hilang diimputasi dengan median untuk fitur numerik dan nilai
tersering untuk fitur kategorikal. Fitur kategorikal kemudian diubah dengan
one-hot encoding.

## Algoritma dan evaluasi

- **Decision Tree** membentuk aturan keputusan dari fitur data.
- **Random Forest** menggabungkan prediksi dari sejumlah decision tree.
- **Dummy Classifier** menjadi baseline dengan selalu memprediksi kelas
  mayoritas pada data training.

Dataset dibagi menjadi 80% training dan 20% testing dengan stratifikasi kelas.
Random state ditetapkan agar pembagian data dan eksperimen dapat direproduksi.
Precision, Recall, dan F1-score menghitung kelas positif **Kualitas Tinggi**.
Model terbaik dipilih dari semua model, termasuk baseline, berdasarkan
F1-score dengan Accuracy sebagai pemecah nilai seri. Dengan demikian,
perbandingan dapat menunjukkan secara jujur apabila model ML belum
mengungguli baseline.

### Hasil eksperimen

Pada eksekusi dengan 1.339 baris data, random state 42, dan pembagian data
80:20, diperoleh hasil berikut (Precision, Recall, dan F1-score untuk kelas
positif **Kualitas Tinggi**):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Dummy Classifier | 0,862 | 0,862 | 1,000 | 0,926 |
| Decision Tree | 0,780 | 0,884 | 0,857 | 0,870 |
| Random Forest | 0,843 | 0,876 | 0,952 | 0,913 |

Pada split ini, Dummy Classifier memperoleh F1-score kelas positif tertinggi
karena selalu memprediksi kelas mayoritas. Hasil tersebut tidak berarti
baseline mengenali kedua kelas dengan baik: recall kelas **Di Bawah Ambang**
adalah 0,000. Sebaliknya, Decision Tree dan Random Forest memiliki recall
kelas minoritas masing-masing 0,30 dan 0,16. Karena itu, accuracy dan
F1-score kelas positif perlu dibaca bersama laporan per kelas dan confusion
matrix. Angka dapat berbeda jika versi data atau konfigurasi eksperimen
berubah; file output dari eksekusi proyek menjadi hasil rujukan.

## Cara menjalankan

Pastikan Python sudah terpasang. Dari terminal di folder proyek:

```powershell
python -m venv .venv
```

Aktifkan environment pada Windows:

```powershell
.venv\Scripts\activate
```

Instal dependensi dan jalankan program:

```powershell
python -m pip install -r requirements.txt
python analisis_kopi.py
```

## Output

Program membuat folder `output/` dan menghasilkan:

- `evaluasi_model.csv`: metrik evaluasi untuk tiga model.
- `hasil_prediksi.csv`: label aktual dan prediksi model terbaik untuk data uji.
- `laporan_klasifikasi.txt`: laporan precision, recall, F1-score, dan support.
- `confusion_matrix.png`: confusion matrix model terbaik.
- `perbandingan_model.png`: grafik perbandingan metrik semua model.

## Keterbatasan

1. Ambang skor 80 adalah aturan eksperimen, bukan kategori kualitas resmi.
2. Dataset dan penilaian yang digunakan belum tentu mewakili semua varietas,
   wilayah, panen, dan kondisi biji kopi.
3. Model memakai atribut yang tercatat di dataset dan tidak memeriksa biji
   secara langsung; hasilnya bukan jaminan kualitas fisik atau rasa.
4. Kinerja pada data uji tidak menjamin kinerja yang sama pada data baru.
5. Skor evaluasi harus dilaporkan dari hasil eksekusi program, bukan
   diasumsikan sebelum pengujian.