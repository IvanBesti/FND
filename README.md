# 📰 Fake News Detector

A web application for detecting fake news using AI and Natural Language Processing. Built with Streamlit for easy deployment and optimized for the LIAR dataset.

## 🌟 Key Features

- ✅ **Fake News Detection**: Analyzes news text using machine learning
- 📊 **Confidence Scores**: Shows prediction confidence levels (51.5% - 92%)
- 🚨 **Suspicious Keyword Detection**: Identifies words commonly found in fake news
- 📈 **Model Statistics**: Displays performance and accuracy metrics
- 🎨 **Modern UI**: User-friendly and responsive interface
- 🌐 **Deploy Ready**: Ready for deployment on Streamlit Cloud
- 🔬 **Real Dataset Trained**: Uses actual LIAR dataset (12.8K samples)

## 🛠️ Technology Stack

- **Framework**: Streamlit
- **Machine Learning**: scikit-learn (TF-IDF + Logistic Regression)
- **NLP**: NLTK for text preprocessing
- **Dataset**: LIAR Dataset (UC Santa Barbara) via PolitiFact.com
- **Features**: 15K TF-IDF features + 8 advanced features
- **Language**: English (matching LIAR dataset language)

## 📁 Project Structure

```
FND/
├── app.py                 # Main Streamlit application
├── model_trainer.py       # Advanced model training script
├── utils.py              # Utility functions
├── quick_test.py         # Model testing script
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── models/              # Trained models folder
│   ├── fake_news_model_advanced.pkl  # Main model (393KB)
│   └── fake_news_model_rf.pkl       # Random Forest model (121MB)
└── data/               # LIAR dataset folder
    ├── train.tsv       # Training data (2.4MB)
    ├── test.tsv        # Test data (294KB)
    └── valid.tsv       # Validation data (294KB)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Model Performance (Real Dataset Results)

The model is trained on the actual LIAR dataset with realistic performance metrics:

### Dataset Information
- **Source**: LIAR Dataset (UC Santa Barbara)
- **Size**: 12,791 political statements from PolitiFact.com
- **Real News**: 7,134 samples (true, mostly-true, half-true)
- **Fake News**: 5,657 samples (pants-fire, false, barely-true)
- **Language**: English

### Model Performance
- **Test Accuracy**: 61.7% (realistic for fake news detection)
- **Cross-validation**: 61.6% accuracy
- **Model Size**: 393KB (production-ready)
- **Inference**: Dynamic confidence scores (51.5% - 92%)

### Feature Engineering
1. **Text Preprocessing**: Cleaning, tokenization, stemming
2. **TF-IDF Features**: 15K features with 1-3 grams
3. **Advanced Features**: 8 additional features (word count, punctuation, etc.)
4. **Hyperparameter Tuning**: GridSearchCV optimization

## 🎯 How to Use

### News Input
1. Enter news text in the input area
2. Click "🔍 Analyze News" button
3. Wait for analysis results

### Result Interpretation
- **✅ LIKELY REAL NEWS**: News predicted as authentic
- **⚠️ LIKELY FAKE NEWS**: News predicted as fake
- **Confidence Score**: Model certainty level (51.5% - 92%)
- **Suspicious Keywords**: Words commonly found in fake news

### Test Examples Available
- 🔬 Scientific News (Real)
- ⚠️ Medical Clickbait (Fake)
- 📊 Economic News (Real)
- 👽 Conspiracy Theory (Fake)
- 📱 Single Word (e.g., "Breaking")
- 🔍 Short Phrase (e.g., "Shocking news today")

## 🔬 Advanced Model Training

The project includes an advanced training pipeline with the LIAR dataset:

### 1. Download LIAR Dataset

Place the LIAR dataset files in the `data/` folder:
- `train.tsv` (training data)
- `test.tsv` (test data) 
- `valid.tsv` (validation data)

### 2. Train Advanced Model

```bash
python model_trainer.py
```

This will:
- Load and process LIAR dataset
- Create advanced features
- Perform hyperparameter tuning
- Save optimized model to `models/fake_news_model_advanced.pkl`

### 3. Test Model Performance

```bash
python quick_test.py
```

This runs comprehensive tests showing:
- Prediction accuracy on diverse examples
- Confidence score variation
- Keyword detection effectiveness

## 🌐 Deployment to Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/fake-news-detector.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Login with GitHub
3. Click "New app"
4. Select repository and branch
5. Set main file path: `app.py`
6. Click "Deploy"

## 🔍 Suspicious Keyword Detection

The application identifies words commonly found in fake news:

### Keyword Categories
- **Sensational**: breaking, urgent, shocking, unbelievable
- **Emotional**: you won't believe, must read, viral, amazing
- **Authority**: mainstream media, government, big pharma, doctors hate
- **Conspiracy**: hidden truth, exposed, secret, coverup
- **Medical**: miracle cure, natural remedy, conspiracy

### Pattern Recognition
- **Fake Indicators**: ALL CAPS, multiple exclamations (!!!), "MUST READ"
- **Real Indicators**: "according to", "study shows", "data indicates"
- **Emotional Language**: excessive use of emotional words
- **Suspicious Punctuation**: overuse of exclamation marks

## 🔧 Customization

### Adding Suspicious Keywords

Edit the `suspicious_keywords` list in the `FakeNewsDetector` class:

```python
self.suspicious_keywords = [
    'breaking', 'urgent', 'shocking',
    # Add new keywords here
    'new_keyword_1', 'new_keyword_2'
]
```

### Model Customization

For different models, edit the `train_model` function in `model_trainer.py`:

```python
# Replace Logistic Regression with another model
from sklearn.ensemble import RandomForestClassifier
self.model = RandomForestClassifier(n_estimators=200, random_state=42)
```

## 📚 Supported Datasets

### 1. LIAR Dataset (Primary)
- **Description**: Political statement fact-checking dataset
- **Format**: TSV with 14 columns
- **Size**: 12,791 statements
- **Source**: PolitiFact.com via UC Santa Barbara
- **Download**: [LIAR Dataset](https://github.com/thiagorainmaker77/liar_dataset)

### 2. Custom Dataset
CSV format with columns:
- `text`: News text (English)
- `label`: 0 (real) or 1 (fake)

## ⚠️ Important Notes

- **Realistic Performance**: 61.7% accuracy reflects real-world challenges
- **Language Compatibility**: Optimized for English text (LIAR dataset language)
- **Verification Required**: Always verify news from trusted sources
- **Educational Tool**: Use as assistance, not sole truth determiner
- **Continuous Learning**: Model performance can be improved with more data

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Create Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Streamlit**: [streamlit.io](https://streamlit.io)
- **Scikit-learn**: [scikit-learn.org](https://scikit-learn.org)
- **NLTK**: [nltk.org](https://nltk.org)

## 📞 Support

Jika ada pertanyaan atau masalah:
1. Buat issue di GitHub repository
2. Check dokumentasi Streamlit untuk masalah deployment
3. Pastikan semua dependencies terinstall dengan benar

---

**Happy Detecting! 🔍✨** 