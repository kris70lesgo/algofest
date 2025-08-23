import pandas as pd
import numpy as np
from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
import joblib
import os

class PortfolioManager:
    def __init__(self, symbols=['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']):
        self.symbols = symbols
        self.models = {}
        self.data = {}
        self.portfolio_weights = {}
        
    def load_all_data(self):
        """Load and process data for all symbols"""
        print("🔄 Loading multi-asset data...")
        
        # Fetch fresh data if needed
        fetcher = DataFetcher(self.symbols)
        engineer = FeatureEngineer()
        
        for symbol in self.symbols:
            try:
                # Load processed data
                data_path = f'data/processed/{symbol}_processed.csv'
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                    self.data[symbol] = df
                    print(f"✅ {symbol}: {len(df)} rows loaded")
                else:
                    print(f"⚠️  {symbol}: Processed data not found, using sample data")
                    # Create sample data for demo
                    dates = pd.date_range('2023-01-01', '2025-08-20', freq='D')[:500]
                    sample_data = pd.DataFrame({
                        'Close': np.random.randn(len(dates)).cumsum() + 200,
                        'RSI': np.random.uniform(20, 80, len(dates)),
                        'MACD': np.random.randn(len(dates)),
                        'Target': np.random.choice([-1, 0, 1], len(dates))
                    }, index=dates)
                    self.data[symbol] = sample_data
                    print(f"✅ {symbol}: {len(sample_data)} sample rows created")
                    
            except Exception as e:
                print(f"❌ Error loading {symbol}: {e}")
    
    def calculate_correlation_matrix(self):
        """Calculate correlation between assets"""
        returns_data = {}
        
        for symbol in self.symbols:
            if symbol in self.data:
                returns_data[symbol] = self.data[symbol]['Close'].pct_change().dropna()
        
        if returns_data:
            correlation_df = pd.DataFrame(returns_data)
            correlation_matrix = correlation_df.corr()
            print("\n📊 Asset Correlation Matrix:")
            print(correlation_matrix.round(3))
            return correlation_matrix
        return None
    
    def optimize_portfolio_weights(self, method='equal_weight'):
        """Calculate optimal portfolio weights"""
        n_assets = len(self.symbols)
        
        if method == 'equal_weight':
            # Simple equal weighting
            weight = 1.0 / n_assets
            self.portfolio_weights = {symbol: weight for symbol in self.symbols}
            
        elif method == 'inverse_volatility':
            # Weight inversely proportional to volatility
            volatilities = {}
            for symbol in self.symbols:
                if symbol in self.data:
                    returns = self.data[symbol]['Close'].pct_change().dropna()
                    volatilities[symbol] = returns.std()
            
            # Calculate inverse volatility weights
            inv_vols = {k: 1/v for k, v in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            self.portfolio_weights = {k: v/total_inv_vol for k, v in inv_vols.items()}
        
        print(f"\n⚖️  Portfolio Weights ({method}):")
        for symbol, weight in self.portfolio_weights.items():
            print(f"{symbol}: {weight:.1%}")
            
        return self.portfolio_weights
    
    def generate_portfolio_signals(self):
        """Generate trading signals for entire portfolio"""
        portfolio_signals = {}
        
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
                
            data = self.data[symbol]
            signals = pd.Series(0, index=data.index)
            
            # Simple technical strategy for each asset
            for i in range(20, len(data)):
                rsi = data['RSI'].iloc[i] if 'RSI' in data.columns else 50
                
                # Buy oversold, sell overbought
                if rsi < 30:
                    signals.iloc[i] = 1  # BUY
                elif rsi > 70:
                    signals.iloc[i] = -1  # SELL
            
            portfolio_signals[symbol] = signals
            buy_count = (signals == 1).sum()
            sell_count = (signals == -1).sum()
            print(f"{symbol}: {buy_count} BUY, {sell_count} SELL signals")
        
        return portfolio_signals
    
    def backtest_portfolio(self, initial_capital=50000):
        """Backtest the multi-asset portfolio"""
        print("\n🔄 Running Portfolio Backtest...")
        
        # Get signals for all assets
        portfolio_signals = self.generate_portfolio_signals()
        
        # Initialize portfolio tracking
        total_value = initial_capital
        individual_values = {symbol: initial_capital * weight 
                           for symbol, weight in self.portfolio_weights.items()}
        
        portfolio_history = []
        
        # Get common date range
        common_dates = None
        for symbol in self.symbols:
            if symbol in self.data:
                if common_dates is None:
                    common_dates = self.data[symbol].index
                else:
                    common_dates = common_dates.intersection(self.data[symbol].index)
        
        if common_dates is None or len(common_dates) == 0:
            print("❌ No common dates found")
            return None
        
        # Simulate portfolio performance
        for date in common_dates[-100:]:  # Last 100 days
            portfolio_value = 0
            
            for symbol in self.symbols:
                if symbol in self.data and date in self.data[symbol].index:
                    current_price = self.data[symbol].loc[date, 'Close']
                    weight = self.portfolio_weights.get(symbol, 0)
                    portfolio_value += total_value * weight
            
            portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value
            })
        
        # Calculate performance metrics
        if portfolio_history:
            start_value = portfolio_history[0]['portfolio_value']
            end_value = portfolio_history[-1]['portfolio_value']
            total_return = (end_value / start_value - 1) * 100
            
            returns = pd.Series([p['portfolio_value'] for p in portfolio_history])
            volatility = returns.pct_change().std() * np.sqrt(252) * 100
            sharpe = returns.pct_change().mean() / returns.pct_change().std() * np.sqrt(252)
            
            results = {
                'Total Return': f"{total_return:.2f}%",
                'Volatility': f"{volatility:.2f}%",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Final Value': f"${end_value:,.2f}",
                'Assets': len(self.symbols)
            }
            
            return results, portfolio_history
        
        return None, None
    
    def run_portfolio_analysis(self):
        """Complete portfolio analysis"""
        print("🏦 MULTI-ASSET PORTFOLIO ANALYSIS")
        print("="*50)
        
        # Load data
        self.load_all_data()
        
        # Calculate correlations
        correlation_matrix = self.calculate_correlation_matrix()
        
        # Optimize weights
        weights = self.optimize_portfolio_weights('inverse_volatility')
        
        # Run backtest
        results, history = self.backtest_portfolio()
        
        if results:
            print("\n📊 PORTFOLIO PERFORMANCE:")
            print("="*30)
            for metric, value in results.items():
                print(f"{metric:<15}: {value}")
            
            # Save results
            os.makedirs('results', exist_ok=True)
            
            # Save portfolio composition
            portfolio_df = pd.DataFrame([
                {'Symbol': symbol, 'Weight': f"{weight:.1%}"} 
                for symbol, weight in self.portfolio_weights.items()
            ])
            portfolio_df.to_csv('results/portfolio_weights.csv', index=False)
            
            # Save correlation matrix
            if correlation_matrix is not None:
                correlation_matrix.to_csv('results/correlation_matrix.csv')
            
            print("\n✅ Portfolio analysis complete!")
            print("📁 Results saved to 'results/' folder")
            
            return results
        
        return None

if __name__ == "__main__":
    pm = PortfolioManager()
    results = pm.run_portfolio_analysis()
