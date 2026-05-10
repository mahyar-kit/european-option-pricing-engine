# European Option Pricing Engine

Three self-contained methods for pricing European options, implemented from scratch in Python.

## What This Project Demonstrates

- **Derivatives pricing** — closed-form Black-Scholes, Monte Carlo simulation, and Cox-Ross-Rubinstein binomial tree
- **Numerical methods** — convergence analysis, error scaling, and the tradeoffs between stochastic and deterministic approximation
- **Quantitative Python** — vectorised NumPy, SciPy statistics, matplotlib visualisation, and pytest-based validation

## Methods Implemented

| Method | Description | Error rate |
|---|---|---|
| **Black-Scholes** | Closed-form solution to the BS PDE via risk-neutral expectation. Prices calls and puts in microseconds. | Machine precision |
| **Monte Carlo** | Simulates terminal stock price under the risk-neutral measure using the exact GBM solution. Returns price and standard error. | $O(1/\sqrt{N})$ |
| **Binomial tree** | Cox-Ross-Rubinstein recombining lattice. Builds terminal nodes in one vectorised step, rolls back by slicing. | $O(1/N)$ |

## Repository Structure

```
european-option-pricing-engine/
├── pricing.py          # All three pricing functions, fully documented
├── notebook.ipynb      # Walkthrough: derivations, convergence charts, findings
├── tests/
│   ├── __init__.py
│   └── test_pricing.py # 31 pytest tests covering parity, convergence, edge cases
├── requirements.txt    # Pinned dependencies (numpy, scipy, matplotlib, pytest, jupyter)
└── README.md
```

## Installation

```bash
git clone https://github.com/your-username/european-option-pricing-engine.git
cd european-option-pricing-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from pricing import black_scholes, monte_carlo_price, binomial_tree_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

# Closed-form
call = black_scholes(S, K, T, r, sigma, "call")          # 10.4506

# Monte Carlo — returns (price, standard_error)
price, se = monte_carlo_price(S, K, T, r, sigma, "call", n_simulations=100_000, seed=42)

# Binomial tree
price = binomial_tree_price(S, K, T, r, sigma, "call", n_steps=500)

# Vectorised — pass arrays for S, K, or sigma
import numpy as np
calls = black_scholes(np.linspace(80, 120, 50), K, T, r, sigma, "call")
```

Run tests:

```bash
pytest tests/ -v
```

## Key Findings

Monte Carlo error contracts at rate $1/\sqrt{N}$: halving the error requires four times the simulations. The CRR binomial tree converges at rate $1/N$ but with a characteristic odd/even oscillation — the signed error alternates sign as step count increases, caused by the discrete lattice straddling the strike differently depending on parity. At 1,000 steps the binomial price is within \$0.002 of Black-Scholes; at 500,000 simulations Monte Carlo is within \$0.05.

## Limitations

- **European exercise only.** No early-exercise logic; American options require dynamic programming (binomial with early-exercise check, or Longstaff-Schwartz MC).
- **Constant volatility.** A single $\sigma$ cannot match market prices across strikes. Production systems use local or stochastic volatility models (Dupire 1994, Heston 1993).
- **Idealised market.** Assumes no dividends, continuous rebalancing, no transaction costs, and unrestricted short-selling.

## References

- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654.
- Cox, J., Ross, S. and Rubinstein, M. (1979). "Option Pricing: A Simplified Approach." *Journal of Financial Economics*, 7(3), 229–263.
- Hull, J. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
