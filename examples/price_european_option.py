import matplotlib.pyplot as plt
from src.gbm import simulate_gbm
from src.monte_carlo_pricing import price_option_mc
from src.utils import black_scholes_price

# Parameters
S0 = 100      # Initial stock price
K = 100       # Strike price
T = 1.0       # Time to maturity
r = 0.05      # Risk-free interest rate
sigma = 0.2   # Volatility
mu = 0.05     # Drift (same as r in risk-neutral world)
dt = 0.01     # Time step
N_paths = 10000

# Simulate stock paths
t, paths = simulate_gbm(S0, mu, sigma, T, dt, N_paths)

# Monte Carlo prices
mc_call = price_option_mc(paths, K, r, T, option_type="call")
mc_put = price_option_mc(paths, K, r, T, option_type="put")

# Black-Scholes prices
bs_call = black_scholes_price(S0, K, T, r, sigma, option_type="call")
bs_put = black_scholes_price(S0, K, T, r, sigma, option_type="put")

# Output results
print(f"Monte Carlo Call Price: {mc_call:.4f} (Black-Scholes: {bs_call:.4f})")
print(f"Monte Carlo Put Price:  {mc_put:.4f} (Black-Scholes: {bs_put:.4f})")

# Plot a few sample paths
plt.figure(figsize=(10, 5))
for i in range(100):
    plt.plot(t, paths[i], color='gray', alpha=0.2)
plt.title("Simulated GBM Paths")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.grid(True)
plt.tight_layout()
plt.show()
