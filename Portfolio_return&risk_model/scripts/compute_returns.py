import pandas as pd
import numpy as np
from utils.config import price_csv_path

# Load the saved stock info CSV
df = pd.read_csv(price_csv_path, parse_dates=['Date'], index_col='Date')
df.head()

#daily returns
daily_returns = df.pct_change()
daily_returns = daily_returns.dropna()

# Annual mean returns and volatility
mean_returns = daily_returns.mean()
volatility = daily_returns.std()