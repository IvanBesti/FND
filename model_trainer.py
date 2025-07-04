import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class AdvancedFakeNewsTrainer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Advanced feature engineering parameters
        self.max_features = 15000
        self.ngram_range = (1, 3)  # Include trigrams
        self.min_df = 3
        self.max_df = 0.95
    
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s\.\!\?\,]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_liar_dataset(self):
        """Load and process LIAR dataset from TSV files"""
        print("Loading LIAR dataset...")
        
        def load_tsv(filepath):
            """Load TSV file with proper column names"""
            columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 
                      'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 
                      'pants_fire_counts', 'context']
            
            try:
                df = pd.read_csv(filepath, sep='\t', names=columns, header=None)
                return df[['statement', 'label']].rename(columns={'statement': 'text'})
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
        
        # Load all splits
        train_df = load_tsv('data/train.tsv')
        test_df = load_tsv('data/test.tsv')
        valid_df = load_tsv('data/valid.tsv')
        
        if train_df is not None and test_df is not None and valid_df is not None:
            # Combine all data for training (we'll split again later)
            df = pd.concat([train_df, test_df, valid_df], ignore_index=True)
            
            print(f"Original dataset size: {len(df)} samples")
            print(f"Original label distribution:\n{df['label'].value_counts()}")
            
            # Convert to binary classification (fake vs real)
            # Fake: pants-fire, false, barely-true
            # Real: true, mostly-true, half-true
            fake_labels = ['pants-fire', 'false', 'barely-true']
            df['binary_label'] = df['label'].apply(lambda x: 1 if x in fake_labels else 0)
            
            print(f"\nBinary label distribution:")
            print(f"Real news (0): {sum(df['binary_label'] == 0)}")
            print(f"Fake news (1): {sum(df['binary_label'] == 1)}")
            
            return df[['text', 'binary_label']].rename(columns={'binary_label': 'label'})
        
        return None
    
    def create_advanced_features(self, texts):
        """Create advanced text features"""
        features = []
        
        for text in texts:
            text_str = str(text).lower()
            
            # Basic statistics
            word_count = len(text_str.split())
            char_count = len(text_str)
            exclamation_count = text_str.count('!')
            question_count = text_str.count('?')
            caps_ratio = sum(1 for c in text_str if c.isupper()) / len(text_str) if text_str else 0
            
            # Suspicious words count
            suspicious_words = ['breaking', 'urgent', 'shocking', 'unbelievable', 'secret', 'exposed']
            suspicious_count = sum(1 for word in suspicious_words if word in text_str)
            
            # Sentiment indicators
            positive_words = ['good', 'great', 'excellent', 'amazing']
            negative_words = ['bad', 'terrible', 'awful', 'horrible']
            positive_count = sum(1 for word in positive_words if word in text_str)
            negative_count = sum(1 for word in negative_words if word in text_str)
            
            features.append([
                word_count, char_count, exclamation_count, question_count,
                caps_ratio, suspicious_count, positive_count, negative_count
            ])
        
        return np.array(features)
    
    def train_model(self, df, use_advanced_features=True, model_type='logistic'):
        """Train the fake news detection model with advanced features"""
        print("Starting advanced model training...")
        
        # Remove duplicates and NaN values
        df = df.drop_duplicates().dropna()
        
        print(f"Final dataset size: {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Preprocess texts
        print("Preprocessing texts...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts after preprocessing
        df = df[df['processed_text'].str.len() > 0]
        
        X_text = df['processed_text']
        y = df['label']
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train_text)}")
        print(f"Test set size: {len(X_test_text)}")
        
        # Create TF-IDF vectors
        print("Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            min_df=self.min_df,
            stop_words='english'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
        X_test_tfidf = self.vectorizer.transform(X_test_text)
        
        # Add advanced features if requested
        if use_advanced_features:
            print("Adding advanced features...")
            X_train_advanced = self.create_advanced_features(X_train_text)
            X_test_advanced = self.create_advanced_features(X_test_text)
            
            # Combine TF-IDF with advanced features
            from scipy.sparse import hstack
            X_train_combined = hstack([X_train_tfidf, X_train_advanced])
            X_test_combined = hstack([X_test_tfidf, X_test_advanced])
        else:
            X_train_combined = X_train_tfidf
            X_test_combined = X_test_tfidf
        
        # Train model with hyperparameter tuning
        print(f"Training {model_type} model with hyperparameter tuning...")
        
        if model_type == 'logistic':
            # Logistic Regression with GridSearch
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'class_weight': ['balanced', None],
                'max_iter': [1000]
            }
            self.model = GridSearchCV(
                LogisticRegression(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
        
        elif model_type == 'random_forest':
            # Random Forest with GridSearch
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'class_weight': ['balanced', None]
            }
            self.model = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
        
        # Fit the model
        self.model.fit(X_train_combined, y_train)
        
        # Get best parameters
        print(f"Best parameters: {self.model.best_params_}")
        
        # Evaluate model
        train_accuracy = self.model.score(X_train_combined, y_train)
        test_accuracy = self.model.score(X_test_combined, y_test)
        
        print(f"\nModel Performance:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Best CV Score: {self.model.best_score_:.4f}")
        
        # Detailed classification report
        y_pred = self.model.predict(X_test_combined)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'best_cv_score': self.model.best_score_,
            'best_params': self.model.best_params_,
            'classification_report': classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
        }
    
    def predict(self, text):
        """Predict if news is fake or real with confidence score"""
        if self.model is None or self.vectorizer is None:
            return None, None
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Transform to TF-IDF vector
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Add advanced features
        advanced_features = self.create_advanced_features([text])
        
        # Combine features
        from scipy.sparse import hstack
        text_features = hstack([text_tfidf, advanced_features])
        
        # Get prediction and probability
        prediction = self.model.predict(text_features)[0]
        probability = self.model.predict_proba(text_features)[0]
        
        return prediction, max(probability)
    
    def analyze_important_features(self, top_n=20):
        """Analyze most important features"""
        if hasattr(self.model.best_estimator_, 'coef_'):
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Add advanced feature names
            advanced_feature_names = [
                'word_count', 'char_count', 'exclamation_count', 'question_count',
                'caps_ratio', 'suspicious_count', 'positive_count', 'negative_count'
            ]
            all_feature_names = list(feature_names) + advanced_feature_names
            
            coefficients = self.model.best_estimator_.coef_[0]
            
            # Get top features for fake news (positive coefficients)
            fake_indices = np.argsort(coefficients)[-top_n:]
            fake_features = [(all_feature_names[i], coefficients[i]) for i in fake_indices]
            
            # Get top features for real news (negative coefficients)
            real_indices = np.argsort(coefficients)[:top_n]
            real_features = [(all_feature_names[i], coefficients[i]) for i in real_indices]
            
            print(f"\nTop {top_n} features indicating FAKE news:")
            for feature, coef in reversed(fake_features):
                print(f"  {feature}: {coef:.4f}")
            
            print(f"\nTop {top_n} features indicating REAL news:")
            for feature, coef in real_features:
                print(f"  {feature}: {coef:.4f}")
            
            return fake_features, real_features
        else:
            print("Feature importance analysis not available for this model type.")
            return None, None
    
    def save_model(self, filepath):
        """Save trained model and vectorizer"""
        if self.model is None or self.vectorizer is None:
            print("No model to save!")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model.best_estimator_,  # Save best model
            'vectorizer': self.vectorizer,
            'best_params': self.model.best_params_,
            'best_score': self.model.best_score_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Advanced model saved to {filepath}")

def main():
    trainer = AdvancedFakeNewsTrainer()
    
    # Load LIAR dataset
    df = trainer.load_liar_dataset()
    
    if df is not None:
        print("LIAR dataset loaded successfully!")
        
        # Train model with different configurations
        print("\n" + "="*50)
        print("Training Logistic Regression with advanced features...")
        results_lr = trainer.train_model(df, use_advanced_features=True, model_type='logistic')
        
        # Analyze features
        trainer.analyze_important_features()
        
        # Save model
        trainer.save_model("models/fake_news_model_advanced.pkl")
        
        print("\n" + "="*50)
        print("Training Random Forest...")
        trainer_rf = AdvancedFakeNewsTrainer()
        results_rf = trainer_rf.train_model(df, use_advanced_features=True, model_type='random_forest')
        trainer_rf.save_model("models/fake_news_model_rf.pkl")
        
        # Compare results
        print("\n" + "="*50)
        print("COMPARISON OF MODELS:")
        print(f"Logistic Regression - Test Accuracy: {results_lr['test_accuracy']:.4f}")
        print(f"Random Forest - Test Accuracy: {results_rf['test_accuracy']:.4f}")
        
        # Create a simple version for the Streamlit app
        simple_trainer = AdvancedFakeNewsTrainer()
        simple_trainer.train_model(df, use_advanced_features=False, model_type='logistic')
        simple_trainer.save_model("models/fake_news_model.pkl")
        
        print("\nTraining completed successfully!")
        print("Models saved:")
        print("- models/fake_news_model.pkl (for Streamlit app)")
        print("- models/fake_news_model_advanced.pkl (advanced version)")
        print("- models/fake_news_model_rf.pkl (Random Forest version)")
        
    else:
        print("Failed to load LIAR dataset. Please check if the TSV files exist in data/ folder.")

if __name__ == "__main__":
    main() 