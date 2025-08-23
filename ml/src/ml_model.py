import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

class MLTradingModel:
    def __init__(self, data_path='data/processed/AAPL_processed.csv'):
        self.data_path = data_path
        self.model_rf = None
        self.features = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
    
    def load_data(self):
        """Load processed data with technical indicators"""
        print("Loading processed data...")
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} rows of data")
        
        # Remove any remaining NaN values
        df = df.dropna()
        print(f"After cleaning: {len(df)} rows")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != 'Target']
        self.feature_names = feature_columns
        
        X = df[feature_columns]
        y = df['Target']
        
        print(f"Features: {len(feature_columns)}")
        print(f"Target distribution:")
        print(y.value_counts().sort_index())
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data with time series consideration"""
        # For time series, use the last 20% as test set (most recent data)
        split_index = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_index]
        self.X_test = X.iloc[split_index:]
        self.y_train = y.iloc[:split_index]
        self.y_test = y.iloc[split_index:]
        
        print(f"\nData Split:")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Training period: {self.X_train.index[0]} to {self.X_train.index[-1]}")
        print(f"Test period: {self.X_test.index} to {self.X_test.index[-1]}")
    
    def train_random_forest(self, n_estimators=100, max_depth=10):
        """Train Random Forest model"""
        print(f"\nTraining Random Forest...")
        
        self.model_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model_rf.fit(self.X_train, self.y_train)
        print("✅ Random Forest training complete")
        
        return self.model_rf
    
    def evaluate_model(self, model, model_name):
        """Comprehensive model evaluation"""
        print(f"\n{'='*50}")
        print(f"EVALUATION: {model_name}")
        print(f"{'='*50}")
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        print(f"Training Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report (Test Set):")
        print(classification_report(self.y_test, y_pred_test))
        
        # Confusion Matrix
        print(f"\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(self.y_test, y_pred_test)
        print("Predicted:  -1    0    1")
        for i, row in enumerate(cm):
            print(f"Actual {[-1,0,1][i]}: {row}")
        
        # Store results
        self.results[model_name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(self.y_test, y_pred_test, output_dict=True),
            'confusion_matrix': cm
        }
        
        return test_accuracy
    
    def feature_importance_analysis(self, model, top_n=15):
        """Analyze and display feature importance"""
        print(f"\n{'='*50}")
        print(f"FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*50}")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Display top features
        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 40)
        for i, (_, row) in enumerate(feature_importance.head(top_n).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<20} : {row['importance']:.4f}")
        
        # Save feature importance
        feature_importance.to_csv('results/feature_importance.csv', index=False)
        
        return feature_importance
    
    def cross_validation(self, model, cv_folds=5):
        """Perform time series cross-validation"""
        print(f"\n{'='*50}")
        print(f"CROSS-VALIDATION (Time Series)")
        print(f"{'='*50}")
        
        # Use TimeSeriesSplit for proper time series validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=tscv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_scores
    
    def save_model(self, model, filename):
        """Save trained model"""
        filepath = f'results/{filename}'
        joblib.dump(model, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def generate_predictions_sample(self, model, n_samples=10):
        """Generate sample predictions with probabilities"""
        print(f"\n{'='*50}")
        print(f"SAMPLE PREDICTIONS (Last {n_samples} days)")
        print(f"{'='*50}")
        
        # Get last n_samples from test set
        X_sample = self.X_test.tail(n_samples)
        y_actual = self.y_test.tail(n_samples)
        
        # Make predictions
        predictions = model.predict(X_sample)
        probabilities = model.predict_proba(X_sample)
        
        # Display results
        label_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        
        for i, (date, pred, actual) in enumerate(zip(X_sample.index, predictions, y_actual)):
            prob_sell, prob_hold, prob_buy = probabilities[i]
            status = "✅" if pred == actual else "❌"
            
            print(f"{date.strftime('%Y-%m-%d')}: {status} Pred: {label_map[pred]:<4} | "
                  f"Actual: {label_map[actual]:<4} | "
                  f"Prob: S:{prob_sell:.2f} H:{prob_hold:.2f} B:{prob_buy:.2f}")
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("🚀 STARTING ML TRADING MODEL PIPELINE")
        print("="*60)
        
        # Load and prepare data
        X, y = self.load_data()
        self.split_data(X, y)
        
        # Train model
        model = self.train_random_forest()
        
        # Evaluate model
        test_accuracy = self.evaluate_model(model, "Random Forest")
        
        # Feature importance analysis
        self.feature_importance_analysis(model)
        
        # Cross-validation
        self.cross_validation(model)
        
        # Sample predictions
        self.generate_predictions_sample(model)
        
        # Save model
        self.save_model(model, 'rf_trading_model.joblib')
        
        print(f"\n🎯 PIPELINE COMPLETE!")
        print(f"Final Test Accuracy: {test_accuracy:.3f}")
        print(f"Model and results saved to 'results/' directory")
        
        return model, test_accuracy

# Usage
if __name__ == "__main__":
    # Initialize and run the pipeline
    ml_model = MLTradingModel()
    model, accuracy = ml_model.run_complete_pipeline()
    
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Model ready for backtesting!")
