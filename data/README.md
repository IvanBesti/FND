# Data Folder

Letakkan dataset Anda di folder ini untuk training model kustom.

## Format Dataset yang Didukung

### 1. LIAR Dataset
- **Filename**: `liar_dataset.tsv`
- **Format**: TSV (Tab-separated values)
- **Download**: [LIAR Dataset](https://github.com/thiagorainmaker77/liar_dataset)

### 2. FakeNewsNet Dataset
- **Filename**: `fake_news.csv`
- **Format**: CSV
- **Columns**: `text`, `label`
- **Download**: [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

### 3. Custom Dataset
- **Filename**: `custom_dataset.csv`
- **Format**: CSV
- **Required Columns**:
  - `text`: Teks berita
  - `label`: 0 (real news) atau 1 (fake news)

## Contoh Format Custom Dataset

```csv
text,label
"Scientists at MIT published groundbreaking research on renewable energy.",0
"BREAKING: Government hiding alien technology for decades!",1
```

## Catatan

- Pastikan encoding file adalah UTF-8
- Untuk dataset besar, pertimbangkan untuk menggunakan sampling
- Model akan secara otomatis memproses dan membersihkan teks 