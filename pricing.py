import math
import random


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Price a European option using the Black-Scholes analytical formula.

    S: current stock price
    K: strike price
    T: time to expiration in years
    r: risk-free interest rate (annualized)
    sigma: volatility of the underlying (annualized)
    option_type: 'call' or 'put'
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if option_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def monte_carlo(S, K, T, r, sigma, option_type="call", num_simulations=100_000, seed=None):
    """
    Price a European option using Monte Carlo simulation.

    S: current stock price
    K: strike price
    T: time to expiration in years
    r: risk-free interest rate (annualized)
    sigma: volatility of the underlying (annualized)
    option_type: 'call' or 'put'
    num_simulations: number of simulated paths
    seed: optional random seed for reproducibility
    """
    rng = random.Random(seed)

    payoffs = []
    for _ in range(num_simulations):
        z = rng.gauss(0, 1)
        ST = S * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
        if option_type == "call":
            payoffs.append(max(ST - K, 0))
        elif option_type == "put":
            payoffs.append(max(K - ST, 0))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    return math.exp(-r * T) * (sum(payoffs) / num_simulations)


def binomial_tree(S, K, T, r, sigma, option_type="call", num_steps=100):
    """
    Price a European option using the Cox-Ross-Rubinstein binomial tree model.

    S: current stock price
    K: strike price
    T: time to expiration in years
    r: risk-free interest rate (annualized)
    sigma: volatility of the underlying (annualized)
    option_type: 'call' or 'put'
    num_steps: number of time steps in the tree
    """
    dt = T / num_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)  # risk-neutral probability
    discount = math.exp(-r * dt)

    # terminal stock prices and option payoffs
    prices = [S * (u**j) * (d ** (num_steps - j)) for j in range(num_steps + 1)]

    if option_type == "call":
        values = [max(price - K, 0) for price in prices]
    elif option_type == "put":
        values = [max(K - price, 0) for price in prices]
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # backward induction
    for _ in range(num_steps):
        values = [
            discount * (p * values[j + 1] + (1 - p) * values[j])
            for j in range(len(values) - 1)
        ]

    return values[0]


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    for opt in ("call", "put"):
        bs = black_scholes(S, K, T, r, sigma, opt)
        mc = monte_carlo(S, K, T, r, sigma, opt, seed=42)
        bt = binomial_tree(S, K, T, r, sigma, opt)
        print(f"{opt.upper():4s}  BS={bs:.4f}  MC={mc:.4f}  BT={bt:.4f}")
