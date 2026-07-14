"""
backend/market_benchmark.py
===========================
Fetches daily market benchmark data (e.g., S&P 500) via yfinance
and correlates it with the daily average sentiment.
"""
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from backend.database import fetch_enriched_articles, upsert_market_correlation, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_TICKER = "SPY"

def get_daily_returns(days_back: int = 14) -> pd.DataFrame:
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back + 5)
        
        df = yf.download(BENCHMARK_TICKER, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        if df.empty:
            return pd.DataFrame()
            
        df['Return'] = df['Close'].pct_change() * 100.0 # Percentage return
        df.reset_index(inplace=True)
        # Ensure 'Date' column is string
        if 'Date' in df.columns:
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return df[['Date', 'Return']].dropna()
    except Exception as e:
        log.error(f"Failed to fetch market data from yfinance: {e}")
        return pd.DataFrame()

def calculate_correlations():
    log.info("Calculating market sentiment correlation...")
    articles = fetch_enriched_articles()
    
    if not articles:
        log.info("No articles to correlate.")
        return
        
    df_arts = pd.DataFrame(articles)
    if 'published_at' not in df_arts.columns or df_arts['published_at'].isnull().all():
        return
        
    df_arts['Date'] = pd.to_datetime(df_arts['published_at']).dt.strftime('%Y-%m-%d')
    daily_sentiment = df_arts.groupby('Date')['sentiment_score'].mean().reset_index()
    
    market_returns = get_daily_returns(days_back=30)
    
    if market_returns.empty:
        return
        
    # Merge and calculate
    merged = pd.merge(daily_sentiment, market_returns, on='Date', how='inner')
    
    for _, row in merged.iterrows():
        upsert_market_correlation(row['Date'], row['sentiment_score'], float(row['Return']))
        
    if len(merged) > 1:
        correlation = merged['sentiment_score'].corr(merged['Return'])
        log.info(f"Sentiment-Market Correlation (Pearson): {correlation:.3f}")
    else:
        log.info("Not enough overlapping data days to calculate correlation.")

if __name__ == "__main__":
    init_db()
    calculate_correlations()
