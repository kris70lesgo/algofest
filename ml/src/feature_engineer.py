import pandas as pd
import numpy as np
import os

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicators"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

class FeatureEngineer:
    def __init__(self):
        # FIXED: Data is inside src/data/raw/, so use relative path from src/
        self.data_dir = 'data/raw'  # Changed from '../data/raw' to 'data/raw'
        self.output_dir = 'data/processed'  # Changed from '../data/processed'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, symbol):
        """Load stock data for a symbol"""
        filepath = f'{self.data_dir}/{symbol}.csv'
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    def add_technical_indicators(self, data):
        """Add all technical indicators to the dataset"""
        df = data.copy()
        
        # Price-based indicators
        df['RSI'] = calculate_rsi(df['Close'])
        
        # MACD indicators
        macd_line, signal_line, histogram = calculate_macd(df['Close'])
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Histogram'] = histogram
        
        # Moving averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = calculate_ema(df['Close'], 12)
        df['EMA_26'] = calculate_ema(df['Close'], 26)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Momentum indicators
        df['Price_Change_1D'] = df['Close'].pct_change(1)
        df['Price_Change_3D'] = df['Close'].pct_change(3)
        df['Price_Change_7D'] = df['Close'].pct_change(7)
        df['Momentum_3'] = df['Close'] - df['Close'].shift(3)
        df['Momentum_7'] = df['Close'] - df['Close'].shift(7)
        
        # Volume indicators
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Volatility indicators
        df['True_Range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # Price position indicators
        df['High_Low_Pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Close_SMA_Ratio'] = df['Close'] / df['SMA_20']
        
        return df
    
    def create_target_labels(self, data, forward_days=5, threshold=0.02):
        """Create buy/sell/hold labels based on future price movements"""
        df = data.copy()
        
        # Calculate future returns
        future_return = df['Close'].shift(-forward_days) / df['Close'] - 1
        
        # Create labels: 1 = Buy, 0 = Hold, -1 = Sell
        conditions = [
            future_return > threshold,    # Buy signal
            future_return < -threshold,   # Sell signal
        ]
        choices = [1, -1]
        df['Target'] = np.select(conditions, choices, default=0)
        
        # Remove last few rows with no future data
        df = df.iloc[:-forward_days]
        
        return df
    
    def process_symbol(self, symbol):
        """Complete feature engineering pipeline for one symbol"""
        print(f"Processing {symbol}...")
        
        # Load data
        data = self.load_data(symbol)
        print(f"Loaded {len(data)} days of data")
        
        # Add technical indicators
        data_with_features = self.add_technical_indicators(data)
        print(f"Added technical indicators")
        
        # Create target labels
        final_data = self.create_target_labels(data_with_features)
        print(f"Created target labels")
        
        # Remove rows with NaN values
        final_data = final_data.dropna()
        print(f"Final dataset: {len(final_data)} rows")
        
        # Save processed data
        output_path = f'{self.output_dir}/{symbol}_processed.csv'
        final_data.to_csv(output_path)
        print(f"Saved to {output_path}")
        
        return final_data

# Usage example
if __name__ == "__main__":
    fe = FeatureEngineer()
    
    # Process single stock for testing
    aapl_data = fe.process_symbol('AAPL')
    
    # Show sample of engineered features
    print("\nSample of engineered features:")
    feature_cols = ['RSI', 'MACD', 'SMA_20', 'BB_Position', 'Volume_Ratio', 'Target']
    print(aapl_data[feature_cols].tail(10))
    
    # Show target distribution
    print("\nTarget Label Distribution:")
    print(aapl_data['Target'].value_counts())
    print("\nAs percentages:")
    print(aapl_data['Target'].value_counts(normalize=True) * 100)
