# Monte Carlo Option Pricing with Geometric Brownian Motion

This project implements a simulation-based engine to price **European call and put options** using **Monte Carlo methods** with **Geometric Brownian Motion (GBM)** for the underlying asset.

The implementation also compares simulated prices with the analytical **Black-Scholes formula** and provides visualizations of simulated price paths.

---

## Features

- Vectorized simulation of GBM paths using NumPy
- Monte Carlo option pricing engine
- Black-Scholes formula implementation for comparison
- Plotting of sample stock paths using `matplotlib`
- Clean modular structure: `src/`, `tests/`, `examples/`
- Minimal test suite using `pytest`

---

## Example Output

```bash
Monte Carlo Call Price: 10.4602 (Black-Scholes: 10.4506)
Monte Carlo Put Price:  5.5730 (Black-Scholes: 5.5735)
