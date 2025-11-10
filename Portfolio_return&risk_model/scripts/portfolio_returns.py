import numpy as np
from utils.config import trading_days
from scripts.portfolio_weights import weights
from scripts.compute_returns import mean_returns, daily_returns

# Daily returns
portfolio_returns_daily = np.dot(weights, mean_returns)

# Annual returns
portfolio_returns_annual = daily_returns * trading_days
