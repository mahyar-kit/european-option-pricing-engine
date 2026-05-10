"""Three methods for pricing European options:
1. Closed-form Black-Scholes
2. Monte Carlo simulation
3. Cox-Ross-Rubinstein binomial tree
"""

import numpy as np
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type):
    """Price a European option using the closed-form Black-Scholes formula.

    Parameters
    ----------
    S : float or np.ndarray
        Current underlying price. Must be positive.
    K : float or np.ndarray
        Strike price. Must be positive.
    T : float
        Time to expiry in years. Must be non-negative.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float or np.ndarray
        Annualised volatility of the underlying. Must be positive.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float or np.ndarray
        Option price. Same shape as input S / K / sigma.

    Raises
    ------
    ValueError
        If inputs violate positivity constraints or option_type is unrecognised.

    Example
    -------
    >>> black_scholes(100, 100, 1.0, 0.05, 0.2, 'call')
    10.450583572185565
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # --- input validation ---
    if np.any(S <= 0):
        raise ValueError("S must be positive.")
    if np.any(K <= 0):
        raise ValueError("K must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if np.any(sigma < 0):
        raise ValueError("sigma must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    # --- edge case: expiry has passed or is now ---
    if T == 0:
        intrinsic_call = np.maximum(S - K, 0.0)
        intrinsic_put  = np.maximum(K - S, 0.0)
        return intrinsic_call if option_type == "call" else intrinsic_put

    # --- edge case: zero vol → only time value from discounting ---
    if np.all(sigma == 0):
        pv_k = K * np.exp(-r * T)           # present value of strike
        intrinsic_call = np.maximum(S - pv_k, 0.0)
        intrinsic_put  = np.maximum(pv_k - S, 0.0)
        return intrinsic_call if option_type == "call" else intrinsic_put

    # --- Black-Scholes d1 and d2 ---
    # d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # d2 = d1 - σ·√T  (same as [ln(S/K) + (r - σ²/2)·T] / (σ·√T))
    d2 = d1 - sigma * np.sqrt(T)

    # N(·) is the standard normal CDF — risk-neutral probability terms
    if option_type == "call":
        # C = S·N(d1) - K·e^{-rT}·N(d2)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        # P = K·e^{-rT}·N(-d2) - S·N(-d1)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def monte_carlo_price(S, K, T, r, sigma, option_type, n_simulations, seed=None):
    """Price a European option via Monte Carlo simulation under the risk-neutral measure.

    Simulates terminal stock price using the exact GBM solution:
        S_T = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)

    Discounts the expected payoff: price = e^{-rT} * E[max(S_T - K, 0)]

    Parameters
    ----------
    S : float
        Current underlying price. Must be positive.
    K : float
        Strike price. Must be positive.
    T : float
        Time to expiry in years. Must be non-negative.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Annualised volatility. Must be positive.
    option_type : str
        'call' or 'put'.
    n_simulations : int
        Number of independent terminal price draws.
    seed : int or None, optional
        Seed for numpy's random Generator. None gives non-deterministic output.

    Returns
    -------
    price : float
        Monte Carlo estimate of the option price.
    std_error : float
        Standard error of the price estimate: std(payoffs) / sqrt(n_simulations).

    Example
    -------
    >>> price, se = monte_carlo_price(100, 100, 1.0, 0.05, 0.2, 'call', 100_000, seed=42)
    >>> abs(price - 10.4506) < 0.1
    True
    """
    if S <= 0:
        raise ValueError("S must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    rng = np.random.default_rng(seed)                        # modern Generator API — avoids legacy global state

    Z = rng.standard_normal(n_simulations)                   # Z ~ N(0,1), shape (n_simulations,)

    # exact GBM terminal price — no discretisation error
    # drift = (r - 0.5*sigma^2)*T removes Ito correction so E[S_T] = S*e^{rT}
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)                   # max(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)                   # max(K - S_T, 0)

    discounted = np.exp(-r * T) * payoffs                    # PV of each simulated payoff

    price     = discounted.mean()                            # E[e^{-rT} * payoff]
    std_error = discounted.std(ddof=1) / np.sqrt(n_simulations)  # CLT standard error

    return price, std_error


def binomial_tree_price(S, K, T, r, sigma, option_type, n_steps):
    """Price a European option via the Cox-Ross-Rubinstein binomial tree.

    CRR parameterisation ensures the tree recombines and matches the first two
    moments of GBM in each interval dt = T / n_steps:
        u = exp(sigma * sqrt(dt))      # up-move factor
        d = 1 / u                      # down-move factor (tree recombines)
        p = (exp(r*dt) - d) / (u - d) # risk-neutral probability of up move

    Terminal stock prices span S*d^n to S*u^n with n_steps+1 distinct nodes.
    Values are rolled back by one step at a time using vectorised array slicing.

    Parameters
    ----------
    S : float
        Current underlying price. Must be positive.
    K : float
        Strike price. Must be positive.
    T : float
        Time to expiry in years. Must be positive.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Annualised volatility. Must be positive.
    option_type : str
        'call' or 'put'.
    n_steps : int
        Number of time steps in the tree. Larger → more accurate, O(n) time.

    Returns
    -------
    float
        Option price.

    Example
    -------
    >>> abs(binomial_tree_price(100, 100, 1.0, 0.05, 0.2, 'call', 500) - 10.4506) < 0.01
    True
    """
    if S <= 0:
        raise ValueError("S must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")

    dt = T / n_steps                                   # length of each time interval
    u  = np.exp(sigma * np.sqrt(dt))                   # up factor: matched to GBM vol
    d  = 1.0 / u                                       # down factor: 1/u ensures recombination
    p  = (np.exp(r * dt) - d) / (u - d)               # risk-neutral up probability

    # --- terminal stock prices (vectorised, no loop) ---
    # after n_steps, node j has had j up-moves and (n_steps-j) down-moves
    # S_T[j] = S * u^j * d^(n_steps-j) = S * u^(2j - n_steps)
    j   = np.arange(n_steps + 1)                       # j = 0,1,...,n_steps (up-move counts)
    S_T = S * u ** (2 * j - n_steps)                   # shape: (n_steps+1,)

    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)                   # terminal call payoffs
    else:
        V = np.maximum(K - S_T, 0.0)                   # terminal put payoffs

    # --- backward induction (loop over time steps, inner op vectorised) ---
    # V has n_steps+1-i nodes at step i counting back from maturity
    # V[j] = e^{-r*dt} * (p*V[j+1] + (1-p)*V[j])  for j=0,...,i-1
    discount = np.exp(-r * dt)
    for _ in range(n_steps):
        V = discount * (p * V[1:] + (1.0 - p) * V[:-1])   # rolls (n+1,) → (n,)

    return float(V[0])


if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    C = black_scholes(S, K, T, r, sigma, "call")
    P = black_scholes(S, K, T, r, sigma, "put")

    print(f"Call price : {C:.6f}")
    print(f"Put  price : {P:.6f}")

    # Put-call parity: C - P = S - K·e^{-rT}
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    print(f"\nPut-call parity check")
    print(f"  C - P          = {lhs:.6f}")
    print(f"  S - K·exp(-rT) = {rhs:.6f}")
    print(f"  Difference     = {abs(lhs - rhs):.2e}  {'PASS' if abs(lhs - rhs) < 1e-10 else 'FAIL'}")

    # --- Monte Carlo vs Black-Scholes ---
    print("\nMonte Carlo check (n=100,000, seed=42)")
    mc_call, mc_se = monte_carlo_price(S, K, T, r, sigma, "call", 100_000, seed=42)
    mc_put,  pu_se = monte_carlo_price(S, K, T, r, sigma, "put",  100_000, seed=42)

    print(f"  MC call  = {mc_call:.6f}  ±{mc_se:.6f}  (BS = {C:.6f})")
    print(f"  MC put   = {mc_put:.6f}  ±{pu_se:.6f}  (BS = {P:.6f})")

    # within 2 standard errors ↔ ~95% confidence interval contains BS price
    call_ok = abs(mc_call - C) <= 2 * mc_se
    put_ok  = abs(mc_put  - P) <= 2 * pu_se
    print(f"  Call within 2 SE of BS: {'PASS' if call_ok else 'FAIL'}")
    print(f"  Put  within 2 SE of BS: {'PASS' if put_ok  else 'FAIL'}")

    # --- Binomial tree vs Black-Scholes ---
    print("\nBinomial tree check (n_steps=500)")
    bt_call = binomial_tree_price(S, K, T, r, sigma, "call", 500)
    bt_put  = binomial_tree_price(S, K, T, r, sigma, "put",  500)

    print(f"  BT call  = {bt_call:.6f}  (BS = {C:.6f},  diff = {abs(bt_call - C):.6f})")
    print(f"  BT put   = {bt_put:.6f}  (BS = {P:.6f},  diff = {abs(bt_put  - P):.6f})")

    call_ok = abs(bt_call - C) < 0.01
    put_ok  = abs(bt_put  - P) < 0.01
    print(f"  Call within $0.01 of BS: {'PASS' if call_ok else 'FAIL'}")
    print(f"  Put  within $0.01 of BS: {'PASS' if put_ok  else 'FAIL'}")
