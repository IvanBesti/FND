# Models Folder

Folder ini menyimpan model yang sudah dilatih untuk deteksi berita palsu.

## File Model

### `fake_news_model.pkl`
- **Deskripsi**: Model utama untuk deteksi berita palsu
- **Format**: Pickle file
- **Isi**:
  - Trained Logistic Regression model
  - TF-IDF Vectorizer
  - Preprocessing parameters

## Cara Kerja

1. **Automatic Loading**: Aplikasi Streamlit secara otomatis memuat model dari file ini
2. **Fallback**: Jika file tidak ada, aplikasi akan menggunakan sample dataset untuk training
3. **Custom Training**: Gunakan `model_trainer.py` untuk membuat model baru

## Model Architecture

```
Input Text
    ↓
Text Preprocessing (cleaning, tokenization, stemming)
    ↓
TF-IDF Vectorization (10,000 features, 1-2 grams)
    ↓
Logistic Regression Classifier
    ↓
Prediction + Confidence Score
```

## Performance Metrics

Dengan sample dataset:
- **Training Accuracy**: ~85%
- **Test Accuracy**: ~82%
- **Cross-validation**: 5-fold CV

## File Size

- Model file biasanya berukuran 5-50 MB tergantung dataset
- Untuk deployment Streamlit Cloud, pastikan total ukuran repository < 500 MB

## Update Model

Untuk memperbarui model:

```bash
python model_trainer.py
```

Model baru akan menggantikan file yang ada di folder ini. 