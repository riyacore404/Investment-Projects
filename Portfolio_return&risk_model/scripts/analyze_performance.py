import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.compute_returns import daily_returns, mean_returns
from scripts.portfolio_returns import portfolio_returns_annual
from scripts.portfolio_risks import portfolio_vol_annual, cov_matrix
from utils.config import risk_free_rate, trading_days

portfolio_sharpe = (portfolio_returns_annual - risk_free_rate) / portfolio_vol_annual

# Monte Carlo Simulation (Random Portfolios)
num_assets = len(daily_returns.columns)
num_portfolios = 5000

all_returns = []
all_volatility = []

for _ in range(num_portfolios):
    w = np.random.random(num_assets)
    w /= np.sum(w)
    r = np.dot(w, mean_returns) * trading_days
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(trading_days)
    all_returns.append(r)
    all_volatility.append(vol)

all_returns = np.array(all_returns)
all_volatility = np.array(all_volatility)


# Risk-Return Visuslization
def extract_scalar(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values.flatten()[0]
    elif isinstance(x, (np.ndarray, list)):
        return np.array(x).flatten()[0]
    else:
        return float(x)

ret_value = extract_scalar(portfolio_returns_annual)
vol_value = extract_scalar(portfolio_vol_annual)

plt.figure(figsize=(10,6))
plt.scatter(all_volatility, all_returns, c=all_returns/all_volatility, cmap='viridis', marker='o', s=10, alpha=0.7)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(vol_value, ret_value, color='red', marker='*', s=200, label='Your Portfolio')

plt.title('Portfolio Risk vs Return (Random Portfolios)')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Return')

plt.legend()
plt.grid(alpha=0.3)
plt.show()