import pickle
import os
from app import FakeNewsDetector

def load_trained_model():
    """Load the trained model"""
    detector = FakeNewsDetector()
    
    model_path = "models/fake_news_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            detector.model = model_data['model']
            detector.vectorizer = model_data['vectorizer']
            print("✅ Model loaded successfully from trained LIAR dataset!")
            return detector
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    else:
        print("❌ No trained model found!")
        return None

def test_real_time_predictions():
    """Test model with real-time examples"""
    detector = load_trained_model()
    
    if detector is None:
        return
    
    # Test cases - mix of real and potentially fake news
    test_cases = [
        {
            "text": "Scientists at Stanford University published a new study in Nature journal showing climate change impacts on marine ecosystems.",
            "expected": "Real News",
            "category": "Real Scientific News"
        },
        {
            "text": "BREAKING: Government hiding secret documents about alien technology for decades, leaked by whistleblower!",
            "expected": "Fake News",
            "category": "Conspiracy Theory"
        },
        {
            "text": "The Federal Reserve announced a quarter-point interest rate increase following their latest policy meeting in Washington.",
            "expected": "Real News", 
            "category": "Real Economic News"
        },
        {
            "text": "SHOCKING: Doctors HATE this one weird trick that cures all diseases! Big Pharma doesn't want you to know!",
            "expected": "Fake News",
            "category": "Medical Misinformation"
        },
        {
            "text": "President Biden met with European leaders to discuss ongoing support for Ukraine amid the conflict with Russia.",
            "expected": "Real News",
            "category": "Real Political News"
        },
        {
            "text": "URGENT: 5G towers are secretly controlling your mind, scientists confirm in classified study!",
            "expected": "Fake News",
            "category": "Technology Conspiracy"
        },
        {
            "text": "Local health officials report increased vaccination rates following a public awareness campaign in the county.",
            "expected": "Real News",
            "category": "Real Health News"
        },
        {
            "text": "VIRAL: Ancient aliens built the pyramids, new archaeological evidence proves government cover-up!",
            "expected": "Fake News",
            "category": "Historical Conspiracy"
        }
    ]
    
    print("\n" + "="*70)
    print("🧪 TESTING FAKE NEWS DETECTOR WITH REAL-TIME EXAMPLES")
    print("="*70)
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📰 Test {i}: {test_case['category']}")
        print(f"Text: {test_case['text'][:80]}...")
        
        # Get prediction
        prediction, confidence, suspicious_words = detector.predict(test_case['text'])
        
        if prediction == 1:
            predicted_label = "Fake News"
            confidence_desc = f"FAKE with {confidence:.1%} confidence"
            status_emoji = "⚠️"
        else:
            predicted_label = "Real News"
            confidence_desc = f"REAL with {confidence:.1%} confidence"
            status_emoji = "✅"
        
        # Check if prediction is correct
        is_correct = predicted_label == test_case['expected']
        if is_correct:
            correct_predictions += 1
            result_emoji = "✅"
        else:
            result_emoji = "❌"
        
        print(f"Expected: {test_case['expected']}")
        print(f"Predicted: {status_emoji} {confidence_desc}")
        print(f"Result: {result_emoji} {'CORRECT' if is_correct else 'INCORRECT'}")
        
        if suspicious_words:
            print(f"🚨 Suspicious keywords: {', '.join(suspicious_words)}")
        else:
            print("✅ No suspicious keywords detected")
    
    # Summary
    accuracy = (correct_predictions / total_tests) * 100
    print("\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)
    print(f"Total Tests: {total_tests}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 75:
        print("🎉 Excellent performance!")
    elif accuracy >= 60:
        print("👍 Good performance!")
    else:
        print("⚠️ Model needs improvement")
    
    return accuracy

def test_suspicious_keywords():
    """Test suspicious keyword detection"""
    detector = load_trained_model()
    
    if detector is None:
        return
    
    print("\n" + "="*70)
    print("🔍 TESTING SUSPICIOUS KEYWORD DETECTION")
    print("="*70)
    
    keyword_tests = [
        "BREAKING news about government conspiracy",
        "You won't believe what scientists discovered",
        "URGENT warning about mainstream media lies",
        "The Federal Reserve announced new policies",
        "Research shows climate change impacts"
    ]
    
    for text in keyword_tests:
        suspicious_words = detector.find_suspicious_keywords(text)
        print(f"\nText: {text}")
        if suspicious_words:
            print(f"🚨 Found: {', '.join(suspicious_words)}")
        else:
            print("✅ No suspicious keywords")

def interactive_test():
    """Interactive testing mode"""
    detector = load_trained_model()
    
    if detector is None:
        return
    
    print("\n" + "="*70)
    print("🎮 INTERACTIVE TESTING MODE")
    print("="*70)
    print("Enter news text to analyze (type 'quit' to exit)")
    
    while True:
        user_input = input("\n📰 Enter news text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            print("⚠️ Please enter some text")
            continue
        
        prediction, confidence, suspicious_words = detector.predict(user_input)
        
        if prediction == 1:
            print(f"⚠️ FAKE NEWS (confidence: {confidence:.1%})")
        else:
            print(f"✅ REAL NEWS (confidence: {confidence:.1%})")
        
        if suspicious_words:
            print(f"🚨 Suspicious keywords: {', '.join(suspicious_words)}")

if __name__ == "__main__":
    print("🔍 FAKE NEWS DETECTOR - MODEL TESTING")
    print("Using LIAR Dataset trained model")
    
    # Run automated tests
    test_real_time_predictions()
    test_suspicious_keywords()
    
    # Ask if user wants interactive mode
    user_choice = input("\n🎮 Do you want to try interactive testing? (y/n): ").strip().lower()
    if user_choice in ['y', 'yes']:
        interactive_test() 