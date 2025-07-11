import numpy as np

def simulate_gbm(S0, mu, sigma, T, dt, N_paths):
    """
    Simulate geometric Brownian motion paths.

    Parameters:
        S0 (float): Initial stock price.
        mu (float): Drift term.
        sigma (float): Volatility.
        T (float): Time to maturity.
        dt (float): Time step size.
        N_paths (int): Number of simulated paths.

    Returns:
        t (ndarray): Time array.
        paths (ndarray): Simulated paths of shape (N_paths, N_steps+1).
    """
    N_steps = int(T / dt)
    t = np.linspace(0, T, N_steps + 1)
    paths = np.zeros((N_paths, N_steps + 1))
    paths[:, 0] = S0

    dW = np.random.normal(0, np.sqrt(dt), size=(N_paths, N_steps))
    W = np.cumsum(dW, axis=1)
    exponent = (mu - 0.5 * sigma**2) * t[1:] + sigma * W
    paths[:, 1:] = S0 * np.exp(exponent)

    return t, paths
