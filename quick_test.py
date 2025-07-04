import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import FakeNewsDetector

def test_improved_predictions():
    """Test the improved prediction logic"""
    print("🔍 Testing Improved Fake News Detection")
    print("=" * 50)
    
    # Load the trained model
    detector = FakeNewsDetector()
    try:
        detector.load_model('models/fake_news_model_advanced.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test cases that should be more accurate
    test_cases = [
        {
            'text': "Breaking news! Scientists discover shocking truth about vaccines!",
            'expected': 1,  # Should be FAKE (sensational, suspicious words)
            'description': "Sensational medical clickbait"
        },
        {
            'text': "According to a study published in Nature, researchers found evidence of climate change effects on marine ecosystems.",
            'expected': 0,  # Should be REAL (scientific, credible source)
            'description': "Academic research news"
        },
        {
            'text': "URGENT! Doctors HATE this simple trick! Big Pharma doesn't want you to know!",
            'expected': 1,  # Should be FAKE (caps, clickbait, conspiracy)
            'description': "Medical conspiracy clickbait"
        },
        {
            'text': "The government announced new economic policies to address inflation concerns during yesterday's press conference.",
            'expected': 0,  # Should be REAL (official statement, factual)
            'description': "Official government announcement"
        },
        {
            'text': "Breaking",
            'expected': 1,  # Should be FAKE (suspicious single word)
            'description': "Single suspicious word"
        },
        {
            'text': "Research shows positive results",
            'expected': 0,  # Should be REAL (research indicator)
            'description': "Research-based phrase"
        },
        {
            'text': "EXPOSED: The hidden truth they don't want you to know about!!!",
            'expected': 1,  # Should be FAKE (conspiracy, caps, exclamations)
            'description': "Conspiracy theory headline"
        },
        {
            'text': "Data shows inflation rates decreased by 2% according to official statistics.",
            'expected': 0,  # Should be REAL (data-driven, official source)
            'description': "Statistical report"
        },
        {
            'text': "SHOCKING: Miracle cure discovered that doctors don't want you to know!",
            'expected': 1,  # Should be FAKE (sensational medical claim)
            'description': "Medical misinformation"
        },
        {
            'text': "The Federal Reserve announced interest rates will remain unchanged following the board meeting.",
            'expected': 0,  # Should be REAL (official financial news)
            'description': "Financial news announcement"
        }
    ]
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {case['description']}")
        print(f"Text: \"{case['text']}\"")
        
        prediction, confidence, suspicious_words = detector.predict(case['text'])
        
        if prediction is None:
            print("❌ Prediction failed")
            continue
            
        result_text = "FAKE" if prediction == 1 else "REAL"
        expected_text = "FAKE" if case['expected'] == 1 else "REAL"
        
        is_correct = prediction == case['expected']
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        
        if is_correct:
            correct_predictions += 1
            
        print(f"Predicted: {result_text} (confidence: {confidence:.1%}) - {status}")
        print(f"Expected: {expected_text}")
        
        if suspicious_words:
            print(f"Suspicious keywords: {', '.join(suspicious_words)}")
        
        print(f"Word count: {len(case['text'].split())}, Suspicious: {len(suspicious_words)}")
    
    print("\n" + "=" * 50)
    accuracy = (correct_predictions / total_tests) * 100
    print(f"📊 FINAL RESULTS:")
    print(f"Correct predictions: {correct_predictions}/{total_tests}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 75:
        print("🎉 EXCELLENT! Model predictions are much improved!")
    elif accuracy >= 62.5:
        print("👍 GOOD! Model predictions are working well!")
    else:
        print("⚠️ Model needs more improvement")

def test_confidence_variation():
    """Test that confidence scores vary appropriately"""
    print("\n🎯 Testing Confidence Score Variation")
    print("=" * 50)
    
    detector = FakeNewsDetector()
    try:
        detector.load_model('models/fake_news_model_advanced.pkl')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test different types of input for confidence variation
    test_inputs = [
        "Breaking",
        "Research shows",
        "URGENT!!! Shocking truth exposed!!!",
        "According to the official study published in Nature journal, scientists have documented significant changes in marine biodiversity patterns over the past decade.",
        "Doctors HATE this simple trick! Big Pharma conspiracy exposed!"
    ]
    
    confidences = []
    
    for text in test_inputs:
        prediction, confidence, suspicious_words = detector.predict(text)
        confidences.append(confidence)
        result = "FAKE" if prediction == 1 else "REAL"
        print(f"Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"Result: {result} (confidence: {confidence:.1%})")
        print(f"Words: {len(text.split())}, Suspicious: {len(suspicious_words)}")
        print()
    
    confidence_range = max(confidences) - min(confidences)
    print(f"Confidence range: {min(confidences):.1%} - {max(confidences):.1%}")
    print(f"Range spread: {confidence_range:.1%}")
    
    if confidence_range > 0.15:
        print("✅ Good confidence variation!")
    else:
        print("⚠️ Low confidence variation")

if __name__ == "__main__":
    test_improved_predictions()
    test_confidence_variation() 