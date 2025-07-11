import numpy as np

def price_option_mc(paths, K, r, T, option_type="call"):
    """
    Price a European option using Monte Carlo simulation.

    Parameters:
        paths (ndarray): Simulated stock price paths.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        T (float): Time to maturity.
        option_type (str): "call" or "put".

    Returns:
        float: Estimated option price.
    """
    ST = paths[:, -1]

    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return np.exp(-r * T) * np.mean(payoff)
