from src.gbm import simulate_gbm
from src.monte_carlo_pricing import price_option_mc
from src.utils import black_scholes_price
import matplotlib.pyplot as plt

def main():
    # --- Parameters ---
    S0 = 100       # Initial stock price
    K = 100        # Strike price
    T = 1.0        # Time to maturity (years)
    r = 0.05       # Risk-free rate
    sigma = 0.2    # Volatility
    mu = r         # Drift = risk-free rate (risk-neutral)
    dt = 0.01      # Time step
    N_paths = 10000
    N_plot_paths = 100  # Number of paths to visualize

    # --- Simulate GBM ---
    t, paths = simulate_gbm(S0, mu, sigma, T, dt, N_paths)

    # --- Price via Monte Carlo ---
    mc_call = price_option_mc(paths, K, r, T, option_type="call")
    mc_put = price_option_mc(paths, K, r, T, option_type="put")

    # --- Price via Black-Scholes ---
    bs_call = black_scholes_price(S0, K, T, r, sigma, option_type="call")
    bs_put = black_scholes_price(S0, K, T, r, sigma, option_type="put")

    # --- Print Results ---
    print("=== European Option Prices ===")
    print(f"Call (MC):  {mc_call:.4f} | Call (BS): {bs_call:.4f}")
    print(f"Put  (MC):  {mc_put:.4f} | Put  (BS): {bs_put:.4f}")

    # --- Plot Sample Paths ---
    plt.figure(figsize=(10, 5))
    for i in range(min(N_plot_paths, N_paths)):
        plt.plot(t, paths[i], color='gray', alpha=0.2)
    plt.title("Sample Simulated GBM Paths")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
