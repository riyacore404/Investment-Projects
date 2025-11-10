import numpy as np
from scripts.compute_returns import daily_returns, volatility, mean_returns

num_assets = len(daily_returns.columns)
equal_weigths = np.ones(num_assets) / num_assets # n_assets = num of assests

# a) Volatility-based weights
inv_vol = 1 / volatility
vol_weights = inv_vol / inv_vol.sum()

# b) Sharpe-based weights
sharpe_ratio = mean_returns / volatility
sharpe_weights = sharpe_ratio / sharpe_ratio.sum()

# Choose weights for analysis
weights = sharpe_weights.values  # or vol_weights.values