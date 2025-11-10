import numpy as np
from utils.config import trading_days
from scripts.portfolio_weights import weights
from scripts.compute_returns import daily_returns

cov_matrix = daily_returns.cov()

# Daily portfolio volatility
portfolio_vol_daily = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Annualized volatility (optional)
portfolio_vol_annual = portfolio_vol_daily * np.sqrt(trading_days)