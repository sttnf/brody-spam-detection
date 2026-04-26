# 🛡️ Brody Spam Shield

> **Sistem deteksi spam & judi online otomatis** berbasis Hybrid Ensemble (Machine Learning & Transformer). Dirancang khusus untuk memfilter ekosistem komentar YouTube Indonesia.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/rafaalrazzak/spam-detection)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Ringkasan Project
Project ini dikembangkan untuk mengatasi maraknya spam "Judi Online" (Judol) pada kolom komentar live stream, khususnya pada channel **Windah Basudara**. Kami menggabungkan 8 arsitektur model berbeda untuk mendapatkan *ensemble verdict* yang akurat.

## 👥 Tim Pengembang (Kelompok Brody)
| Nama | NIM |
| :--- | :---: |
| **Rafa Al Razzak** | 0110224155 |
| **Muhamad Fadil** | 0110224112 | 
| **Raffa Yuda Pratama** | 0110224081 |
| **Aan Adriana** | 0110224014 |
| **Oryza Ayunda Putri** | 0110224030 |

## 🤖 Model Intelligence
Sistem ini menggunakan 8 arsitektur model yang bekerja secara paralel:

| Kategori | Model | Fitur | Keunggulan |
| :--- | :--- | :---: | :--- |
| **Transformer** | **IndoBERT Fine-Tuned** | Raw Text | **Akurasi Tertinggi.** Memahami konteks slang Indonesia. |
| **ML Klasik** | **XGBoost** | TF-IDF | Menangkap pola kata kunci spam yang berulang. |
| **ML Klasik** | **Random Forest** | TF-IDF | Mengurangi *false positive* melalui teknik ensemble tree. |
| **ML Klasik** | **Linear SVM** | TF-IDF | Handal dalam menangani klasifikasi teks dimensi tinggi. |
| **ML Klasik** | **Complement NB** | TF-IDF | Dioptimalkan untuk dataset yang tidak seimbang (*imbalanced*). |
| **ML Klasik** | **Logistic Regression** | TF-IDF | Model baseline yang cepat dan memiliki interpretasi tinggi. |
| **Semantic ML** | **LightGBM** | Word2Vec | Efisien dalam memproses hubungan makna antar kata. |
| **Semantic ML** | **Logistic Regression** | Word2Vec | Menangkap kemiripan makna teks meskipun kata berbeda. |

## ⚙️ Ensemble Logic
Aplikasi memproses setiap komentar melalui 4 tahapan:
1. **Preprocessing**: Pembersihan teks dan normalisasi karakter.
2. **Feature Extraction**: Konversi teks menjadi vektor (TF-IDF & Word2Vec).
3. **Parallel Prediction**: Seluruh model memberikan probabilitas secara simultan.
4. **Majority Voting**: Keputusan akhir diambil berdasarkan kesepakatan mayoritas model.

## 📊 Dataset, Laporan dan Model
* **Dataset Utama**: [`dataset/windah.csv`](dataset/windah.csv) (5,000+ data berlabel manual).
* **Laporan Teknis**: [`docs/reports.pdf`](docs/reports.pdf) (Detail metodologi & evaluasi).
* **Model Assets**: [Hugging Face Files & Models](https://huggingface.co/spaces/rafaalrazzak/spam-detection/tree/main)

## 🚀 Instalasi Lokal

1. **Clone Repository**
   ```bash
   git clone [https://github.com/sttnf/brody-spam-detection.git](https://github.com/sttnf/brody-spam-detection.git)
   cd brody-spam-detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application**
   ```bash
   python app.py
   ```

## 🌐 Demo
👉 **[Live Demo: Windah Spam Shield](https://huggingface.co/spaces/rafaalrazzak/spam-detection)**

---
*Project ini merupakan bagian dari tugas akademik Sekolah Tinggi Teknologi Terpadu Nurul Fikri (STT-NF) 2026.*
