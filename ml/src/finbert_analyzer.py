import os
import warnings
from transformers import pipeline
import pandas as pd
import numpy as np
from typing import Dict, List, Union

# Suppress warnings for cleaner output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

class FinBERTSentimentAnalyzer:
    def __init__(self):
        """Initialize reliable sentiment analyzer using transformers pipeline"""
        print("🔄 Loading reliable sentiment analysis pipeline...")
        
        # Try multiple models in order of reliability
        models_to_try = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",  # Most reliable
            "nlptown/bert-base-multilingual-uncased-sentiment",  # Backup option 1
            "ProsusAI/finbert"  # Original FinBERT as last resort
        ]
        
        self.sentiment_pipeline = None
        self.model_name = None
        
        for model_name in models_to_try:
            try:
                print(f"Trying model: {model_name}")
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=model_name,
                    return_all_scores=True
                )
                self.model_name = model_name
                print(f"✅ Successfully loaded: {model_name}")
                break
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        if self.sentiment_pipeline is None:
            raise Exception("❌ Could not load any sentiment analysis model")
    
    def analyze_sentiment(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Analyze sentiment using reliable transformers pipeline
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Sentiment analysis results
        """
        if isinstance(text, str):
            texts = [text]
            single_text = True
        else:
            texts = text
            single_text = False
        
        results = []
        
        for txt in texts:
            try:
                # Get all sentiment scores
                pipeline_result = self.sentiment_pipeline(txt)[0]
                
                # Convert pipeline output to standardized format
                sentiment_scores = {}
                for item in pipeline_result:
                    label = item['label'].lower()
                    score = float(item['score'])
                    
                    # Normalize label names
                    if label in ['negative', 'neg']:
                        sentiment_scores['negative'] = score
                    elif label in ['positive', 'pos']:
                        sentiment_scores['positive'] = score
                    elif label in ['neutral', 'neu']:
                        sentiment_scores['neutral'] = score
                
                # Ensure all three categories exist
                if 'negative' not in sentiment_scores:
                    sentiment_scores['negative'] = 0.0
                if 'neutral' not in sentiment_scores:
                    sentiment_scores['neutral'] = 0.0
                if 'positive' not in sentiment_scores:
                    sentiment_scores['positive'] = 0.0
                
                # Determine overall sentiment
                max_label = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k])
                sentiment_score = sentiment_scores['positive'] - sentiment_scores['negative']
                
                result = {
                    'text': txt[:100] + '...' if len(txt) > 100 else txt,
                    'label': max_label,
                    'probabilities': sentiment_scores,
                    'sentiment_score': sentiment_score,
                    'confidence': sentiment_scores[max_label]
                }
                results.append(result)
                
            except Exception as e:
                print(f"⚠️ Error analyzing text: {e}")
                # Return neutral result as fallback
                result = {
                    'text': txt[:100] + '...' if len(txt) > 100 else txt,
                    'label': 'neutral',
                    'probabilities': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33},
                    'sentiment_score': 0.0,
                    'confidence': 0.34
                }
                results.append(result)
        
        return results[0] if single_text else results
    
    def get_aggregated_sentiment(self, texts: List[str]) -> Dict:
        """Get aggregated sentiment from multiple texts"""
        if not texts:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'count': 0}
        
        results = self.analyze_sentiment(texts)
        sentiment_scores = [r['sentiment_score'] for r in results]
        
        return {
            'sentiment_score': np.mean(sentiment_scores),
            'sentiment_std': np.std(sentiment_scores),
            'positive_ratio': sum(1 for r in results if r['label'] == 'positive') / len(results),
            'negative_ratio': sum(1 for r in results if r['label'] == 'negative') / len(results),
            'neutral_ratio': sum(1 for r in results if r['label'] == 'neutral') / len(results),
            'confidence': np.mean([r['confidence'] for r in results]),
            'count': len(results)
        }

# 🧪 COMPREHENSIVE VALIDATION TEST
if __name__ == "__main__":
    print("🧪 RUNNING COMPREHENSIVE SENTIMENT VALIDATION")
    print("="*60)
    
    # Initialize analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    # Test cases with CLEAR expected results
    validation_cases = [
        # POSITIVE cases
        ("Apple reports record profits and beats all expectations", "positive"),
        ("Stock price surges 20% on excellent quarterly results", "positive"),
        ("Company announces dividend increase and share buyback", "positive"),
        ("Tesla delivers record number of vehicles this quarter", "positive"),
        ("Strong earnings drive investor optimism", "positive"),
        
        # NEGATIVE cases  
        ("Company files for bankruptcy amid mounting losses", "negative"),
        ("Stock crashes 30% on disappointing earnings miss", "negative"),
        ("Massive layoffs announced as revenue plummets", "negative"),
        ("Credit rating downgraded to junk status", "negative"),
        ("CEO resigns following fraud investigation", "negative"),
        
        # NEUTRAL cases
        ("Company reports quarterly results in line with estimates", "neutral"),
        ("Stock price remains unchanged in light trading", "neutral"),
        ("No major developments reported this quarter", "neutral")
    ]
    
    print(f"Testing {len(validation_cases)} cases with model: {analyzer.model_name}")
    print("-" * 60)
    
    correct_predictions = 0
    total_predictions = len(validation_cases)
    
    for i, (text, expected) in enumerate(validation_cases, 1):
        result = analyzer.analyze_sentiment(text)
        predicted = result['label']
        score = result['sentiment_score']
        confidence = result['confidence']
        
        is_correct = predicted == expected
        if is_correct:
            correct_predictions += 1
        
        status_emoji = "✅" if is_correct else "❌"
        
        print(f"{status_emoji} Test {i:2d}: Expected {expected.upper():<8} | "
              f"Got {predicted.upper():<8} | Score: {score:+.3f} | Conf: {confidence:.3f}")
        print(f"    Text: {text}")
        print()
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    
    print("🎯 FINAL RESULTS:")
    print("="*40)
    print(f"Model Used: {analyzer.model_name}")
    print(f"Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("🟢 EXCELLENT: Ready for trading integration!")
    elif accuracy >= 60:
        print("🟡 ACCEPTABLE: Usable but could be improved")
    else:
        print("🔴 POOR: Need to try different approach")
    
    print("\n" + "="*60)
    
    # Test aggregated sentiment
    print("🔍 TESTING AGGREGATED SENTIMENT:")
    
    positive_news = [
        "Apple beats earnings expectations",
        "Stock hits new all-time high",
        "Strong quarterly results announced"
    ]
    
    negative_news = [
        "Company reports major losses", 
        "Stock falls on bad news",
        "Disappointing earnings results"
    ]
    
    pos_result = analyzer.get_aggregated_sentiment(positive_news)
    neg_result = analyzer.get_aggregated_sentiment(negative_news)
    
    print(f"Positive News Sentiment: {pos_result['sentiment_score']:+.3f}")
    print(f"Negative News Sentiment: {neg_result['sentiment_score']:+.3f}")
    
    if pos_result['sentiment_score'] > 0 and neg_result['sentiment_score'] < 0:
        print("✅ Aggregated sentiment working correctly!")
    else:
        print("❌ Aggregated sentiment still has issues")
