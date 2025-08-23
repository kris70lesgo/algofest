import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class MarketRegimeDetector:
    REGIME_NAMES = {
        0: "📈  BULL MARKET",
        1: "📉  BEAR MARKET", 
        2: "📊  SIDEWAYS MARKET"
    }

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        feats["trend"] = df["Close"].rolling(20).apply(lambda x: np.polyfit(range(20), x, 1)[0] if len(x)==20 else 0)
        ret = df["Close"].pct_change()
        feats["volatility"] = ret.rolling(20).std() * np.sqrt(252)
        feats["momentum"] = df["Close"].pct_change(20).rolling(10).mean()
        if "Volume" in df.columns:
            feats["volume_trend"] = df["Volume"].rolling(20).mean().pct_change(10)
        else:
            feats["volume_trend"] = 0
        feats["fear_index"] = feats["volatility"].rolling(10).std()
        return feats.dropna()

    def _cluster_to_regime(self, features):
        cols = ['trend', 'volatility', 'momentum', 'volume_trend', 'fear_index']
        X = features[cols].dropna()
        if len(X) < 10:
            print("⚠️  Insufficient data for regime detection")
            return pd.Series(1, index=features.index)

        X_norm = (X - X.mean()) / X.std()
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_norm)

        # Calculate stats for each cluster - keys are integers (0,1,2)
        stats = {}
        for i in range(3):
            mask = labels == i
            if mask.sum() > 0:
                stats[i] = X.loc[mask, 'trend'].mean()

        # Sort clusters by trend - this gives us integers, not lists
        sorted_clusters = sorted(stats.keys(), key=stats.get)

        # Map clusters to regimes - all keys are integers
        mapping = {
            sorted_clusters[2]: 0,  # highest trend = Bull
            sorted_clusters: 1,  # lowest trend = Bear  
            sorted_clusters[1]: 2   # middle = Sideways
        }

        return pd.Series([mapping.get(label, 1) for label in labels], index=X.index)

    def run_regime_analysis(self, symbol="AAPL"):
        print("🌍 MARKET REGIME ANALYSIS")
        print("========================================")

        fpath = f"data/processed/{symbol}_processed.csv"
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        else:
            print(f"⚠️ {fpath} missing – generating sample data.")
            dates = pd.date_range("2023-01-01", "2025-08-20", freq="D")[:400]
            df = pd.DataFrame({
                "Close": np.random.randn(len(dates)).cumsum() + 200,
                "Volume": np.random.randint(1_000_000, 10_000_000, len(dates))
            }, index=dates)

        print(f"📊 {symbol}: {len(df)} rows")

        features = self._build_features(df)
        regimes = self._cluster_to_regime(features)

        curr = regimes.iloc[-1]
        print(f"\n🎯 CURRENT REGIME: {self.REGIME_NAMES[curr]}\n")

        print("📈 REGIME DISTRIBUTION:")
        for r_id, count in regimes.value_counts().sort_index().items():
            pct = count / len(regimes) * 100
            print(f"{self.REGIME_NAMES[r_id]:<22}: {count:>4} days  ({pct:4.1f}%)")

        ret = df["Close"].pct_change()
        print("\n💰 PERFORMANCE BY REGIME:")
        stats = {}
        for r_id, r_name in self.REGIME_NAMES.items():
            mask = regimes == r_id
            if mask.sum() == 0:
                continue
            sub = ret[mask]
            ann_ret = sub.mean() * 252 * 100
            ann_vol = sub.std() * np.sqrt(252) * 100
            sharpe = (sub.mean() / sub.std()) * np.sqrt(252) if sub.std() > 0 else 0
            stats[r_name] = {"days": mask.sum(), "ret": ann_ret, "vol": ann_vol, "sharpe": sharpe}
            print(f"{r_name}:  {ann_ret:+6.1f}%  |  σ {ann_vol:5.1f}%  |  Sharpe {sharpe:4.2f}")

        os.makedirs("results", exist_ok=True)
        pd.DataFrame({
            "Date": regimes.index, 
            "Regime_ID": regimes.values, 
            "Regime_Name": [self.REGIME_NAMES[r] for r in regimes.values]
        }).to_csv("results/market_regimes.csv", index=False)
        print("\n✅ Regimes saved → results/market_regimes.csv")
        
        return {"current_regime": self.REGIME_NAMES[curr], "stats": stats}

if __name__ == "__main__":
    MarketRegimeDetector().run_regime_analysis("AAPL")
