import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ImprovedMLTradingModel:
    def __init__(self, data_path='data/processed/AAPL_processed.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        os.makedirs('results', exist_ok=True)
    
    def load_data(self):
        """Load and preprocess data"""
        print("Loading processed data...")
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        df = df.dropna()
        
        # Remove highly correlated features to reduce overfitting
        feature_columns = [col for col in df.columns if col != 'Target']
        
        # Remove redundant features (keep only best from each category)
        selected_features = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'BB_Position', 'BB_Width',
            'Price_Change_1D', 'Price_Change_3D', 'Price_Change_7D',
            'Volume_Ratio', 'ATR', 'Close_SMA_Ratio'
        ]
        
        # Only use features that exist in the data
        available_features = [f for f in selected_features if f in feature_columns]
        self.feature_names = available_features
        
        X = df[available_features]
        y = df['Target']
        
        print(f"Using {len(available_features)} selected features")
        print(f"Target distribution: {y.value_counts().sort_index().to_dict()}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.3):
        """Split data with larger test set for better validation"""
        split_index = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_index]
        self.X_test = X.iloc[split_index:]
        self.y_train = y.iloc[:split_index]
        self.y_test = y.iloc[split_index:]
        
        # Scale features
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print(f"Training: {len(self.X_train)} samples")
        print(f"Test: {len(self.X_test)} samples")
    
    def train_optimized_model(self):
        """Train model with proper regularization"""
        print("Training optimized Random Forest...")
        
        # Conservative hyperparameters to prevent overfitting
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt', 0.5]
        }
        
        # Use TimeSeriesSplit for proper validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        rf = RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        )
        
        # Grid search with time series CV
        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, 
            scoring='f1_macro', n_jobs=-1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return self.model
    
    def evaluate_model(self):
        """Comprehensive evaluation"""
        print("\n" + "="*50)
        print("IMPROVED MODEL EVALUATION")
        print("="*50)
        
        # Predictions
        y_pred_train = self.model.predict(self.X_train_scaled)
        y_pred_test = self.model.predict(self.X_test_scaled)
        
        # Accuracies
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print(f"Overfitting Gap: {train_acc - test_acc:.3f}")
        
        # Detailed report
        print("\nClassification Report:")
        report = classification_report(self.y_test, y_pred_test, output_dict=True)
        print(classification_report(self.y_test, y_pred_test))
        
        # Trading-specific metrics
        print("\nTrading Signal Performance:")
        print(f"SELL Signal Recall: {report['-1']['recall']:.3f} (% of crashes caught)")
        print(f"BUY Signal Precision: {report['1']['precision']:.3f} (% profitable buys)")
        print(f"HOLD Signal F1: {report['0']['f1-score']:.3f} (risk management)")
        
        return test_acc, report
    
    def feature_importance_analysis(self):
        """Feature importance with interpretability"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE (TOP 10)")
        print("="*50)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<20} : {row['importance']:.4f}")
        
        return importance_df
    
    def run_improved_pipeline(self):
        """Run the improved ML pipeline"""
        print("🔧 IMPROVED ML TRADING MODEL PIPELINE")
        print("="*60)
        
        # Load and prepare data
        X, y = self.load_data()
        self.split_data(X, y)
        
        # Train optimized model
        model = self.train_optimized_model()
        
        # Evaluate
        test_accuracy, report = self.evaluate_model()
        
        # Feature importance
        importance_df = self.feature_importance_analysis()
        
        # Save results
        joblib.dump(model, 'results/improved_rf_model.joblib')
        importance_df.to_csv('results/improved_feature_importance.csv', index=False)
        
        print(f"\n🎯 IMPROVED PIPELINE COMPLETE!")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        
        # Recommendations
        if test_accuracy < 0.55:
            print("\n⚠️  RECOMMENDATIONS:")
            print("1. Consider ensemble methods (XGBoost + Random Forest)")
            print("2. Add more diverse features (sentiment, news, macro indicators)")
            print("3. Optimize threshold for buy/sell signals")
            print("4. Consider different time horizons for predictions")
        
        return model, test_accuracy

# Usage
if __name__ == "__main__":
    improved_model = ImprovedMLTradingModel()
    model, accuracy = improved_model.run_improved_pipeline()
