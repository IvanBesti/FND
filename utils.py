import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
from collections import Counter
import string

# Download required NLTK data
def ensure_nltk_data():
    """Ensure required NLTK data is downloaded"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

class TextAnalyzer:
    """Utility class for text analysis and preprocessing"""
    
    def __init__(self):
        ensure_nltk_data()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Suspicious patterns often found in fake news
        self.suspicious_patterns = {
            'sensational_words': [
                'breaking', 'urgent', 'shocking', 'unbelievable', 'amazing', 'incredible',
                'secret', 'hidden', 'exposed', 'revealed', 'conspiracy', 'leaked'
            ],
            'emotional_triggers': [
                'you wont believe', 'this will shock you', 'must read', 'must see',
                'doctors hate', 'they dont want you to know', 'wake up', 'viral'
            ],
            'authority_undermining': [
                'mainstream media', 'government coverup', 'big pharma', 'establishment',
                'sheeple', 'propaganda', 'fake news', 'hoax', 'scam'
            ],
            'urgency_markers': [
                'urgent', 'breaking', 'immediately', 'now', 'today only', 'limited time'
            ]
        }
    
    def basic_text_stats(self, text):
        """Get basic statistics about the text"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Remove punctuation for word count
        words_no_punct = [word for word in words if word not in string.punctuation]
        
        stats = {
            'char_count': len(text),
            'word_count': len(words_no_punct),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words_no_punct) / len(sentences) if sentences else 0,
            'avg_chars_per_word': np.mean([len(word) for word in words_no_punct]) if words_no_punct else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        }
        
        return stats
    
    def find_suspicious_patterns(self, text):
        """Find suspicious patterns that might indicate fake news"""
        text_lower = text.lower()
        found_patterns = {}
        
        for category, patterns in self.suspicious_patterns.items():
            found = []
            for pattern in patterns:
                if pattern in text_lower:
                    found.append(pattern)
            if found:
                found_patterns[category] = found
        
        return found_patterns
    
    def calculate_readability_score(self, text):
        """Calculate a simple readability score (Flesch Reading Ease approximation)"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        
        if not sentences or not words:
            return 0
        
        # Count syllables (simple approximation)
        syllable_count = 0
        for word in words:
            syllable_count += max(1, len([char for char in word if char in 'aeiou']))
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllable_count / len(words)
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return max(0, min(100, score))  # Clamp between 0-100
    
    def extract_keywords(self, text, top_n=10):
        """Extract top keywords from text using TF-IDF-like approach"""
        # Basic preprocessing
        text_lower = text.lower()
        
        # Remove URLs and special characters
        text_clean = re.sub(r'http\S+|www\S+|https\S+', '', text_lower)
        text_clean = re.sub(r'[^a-zA-Z\s]', '', text_clean)
        
        # Tokenize and remove stopwords
        words = word_tokenize(text_clean)
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Return top keywords
        return word_freq.most_common(top_n)
    
    def analyze_sentiment_indicators(self, text):
        """Analyze sentiment indicators in the text"""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Look for excessive punctuation (potential emotional language)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_words = len([word for word in text.split() if word.isupper() and len(word) > 1])
        
        return {
            'positive_words': positive_count,
            'negative_words': negative_count,
            'exclamation_marks': exclamation_count,
            'question_marks': question_count,
            'caps_words': caps_words,
            'emotional_intensity': (exclamation_count + caps_words) / len(text.split()) if text.split() else 0
        }
    
    def comprehensive_analysis(self, text):
        """Perform comprehensive text analysis"""
        return {
            'basic_stats': self.basic_text_stats(text),
            'suspicious_patterns': self.find_suspicious_patterns(text),
            'readability_score': self.calculate_readability_score(text),
            'keywords': self.extract_keywords(text),
            'sentiment_indicators': self.analyze_sentiment_indicators(text)
        }

def create_feature_vector(text_analysis):
    """Create a feature vector from text analysis for additional model features"""
    stats = text_analysis['basic_stats']
    sentiment = text_analysis['sentiment_indicators']
    
    features = [
        stats['word_count'],
        stats['sentence_count'],
        stats['avg_words_per_sentence'],
        stats['uppercase_ratio'],
        stats['punctuation_ratio'],
        text_analysis['readability_score'],
        sentiment['exclamation_marks'],
        sentiment['caps_words'],
        sentiment['emotional_intensity'],
        len(text_analysis['suspicious_patterns'])  # Number of suspicious pattern categories found
    ]
    
    return np.array(features)

def batch_analyze_texts(texts, analyzer=None):
    """Analyze multiple texts in batch"""
    if analyzer is None:
        analyzer = TextAnalyzer()
    
    results = []
    for text in texts:
        try:
            analysis = analyzer.comprehensive_analysis(text)
            results.append(analysis)
        except Exception as e:
            print(f"Error analyzing text: {e}")
            results.append(None)
    
    return results

def export_analysis_to_csv(analyses, texts, output_file='text_analysis.csv'):
    """Export text analyses to CSV file"""
    data = []
    
    for i, (text, analysis) in enumerate(zip(texts, analyses)):
        if analysis is None:
            continue
            
        row = {
            'text_id': i,
            'text': text[:100] + '...' if len(text) > 100 else text,  # Truncate for readability
            'word_count': analysis['basic_stats']['word_count'],
            'sentence_count': analysis['basic_stats']['sentence_count'],
            'readability_score': analysis['readability_score'],
            'suspicious_pattern_count': len(analysis['suspicious_patterns']),
            'emotional_intensity': analysis['sentiment_indicators']['emotional_intensity'],
            'top_keywords': ', '.join([word for word, freq in analysis['keywords'][:5]])
        }
        
        # Add suspicious patterns
        for category, patterns in analysis['suspicious_patterns'].items():
            row[f'suspicious_{category}'] = ', '.join(patterns)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Analysis exported to {output_file}")
    
    return df

# Example usage functions
def quick_fake_news_check(text):
    """Quick check for fake news indicators"""
    analyzer = TextAnalyzer()
    analysis = analyzer.comprehensive_analysis(text)
    
    # Simple scoring system
    risk_score = 0
    warnings = []
    
    # Check for suspicious patterns
    if analysis['suspicious_patterns']:
        risk_score += len(analysis['suspicious_patterns']) * 20
        warnings.append(f"Found {len(analysis['suspicious_patterns'])} types of suspicious patterns")
    
    # Check emotional intensity
    if analysis['sentiment_indicators']['emotional_intensity'] > 0.1:
        risk_score += 15
        warnings.append("High emotional language detected")
    
    # Check readability (very low or very high might be suspicious)
    readability = analysis['readability_score']
    if readability < 30 or readability > 90:
        risk_score += 10
        warnings.append(f"Unusual readability score: {readability:.1f}")
    
    # Check for excessive uppercase
    if analysis['basic_stats']['uppercase_ratio'] > 0.05:
        risk_score += 10
        warnings.append("Excessive use of uppercase letters")
    
    risk_level = "LOW" if risk_score < 20 else "MEDIUM" if risk_score < 50 else "HIGH"
    
    return {
        'risk_score': min(100, risk_score),
        'risk_level': risk_level,
        'warnings': warnings,
        'analysis': analysis
    } 