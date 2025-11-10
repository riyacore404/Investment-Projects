import pandas as pd
import yfinance as yf
from utils.config import tickers, start, end, price_csv_path

def fetch_prices(tickers=tickers, start=start, end=end, filename=price_csv_path):
    portfolio_data = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            continue
        prices = data["Close"]
        prices.name = ticker
        portfolio_data.append(prices)

    portfolio_df = pd.concat(portfolio_data, axis=1)
    portfolio_df.reset_index(inplace=True)
    
    portfolio_df.to_csv(filename, index=False)
    return portfolio_df

if __name__ == "__main__":
    fetch_prices()