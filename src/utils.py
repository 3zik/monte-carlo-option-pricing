import numpy as np
from scipy.stats import norm

def black_scholes_price(S0, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes price for a European option.

    Parameters:
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        option_type (str): "call" or "put".

    Returns:
        float: Option price.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
