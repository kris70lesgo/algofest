import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, symbols=['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']):
        self.symbols = symbols
        self.data_dir = 'data/raw'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_historical_data(self, period='2y'):
        """Download historical data for all symbols"""
        all_data = {}
        
        for symbol in self.symbols:
            print(f"Downloading {symbol} data...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Save to CSV
            data.to_csv(f'{self.data_dir}/{symbol}.csv')
            all_data[symbol] = data
            print(f"✅ {symbol}: {len(data)} days downloaded")
        
        return all_data
    
    def get_live_data(self, symbol):
        """Get real-time data for a symbol"""
        ticker = yf.Ticker(symbol)
        return ticker.history(period='1d')

# Test the fetcher
if __name__ == "__main__":
    fetcher = DataFetcher()
    data = fetcher.fetch_historical_data()
    print("Data fetching complete!")
