"""Unit tests for pricing.py"""

import numpy as np
import pytest
from pricing import black_scholes, monte_carlo_price, binomial_tree_price


# ---------------------------------------------------------------------------
# Shared parameter sets
# ---------------------------------------------------------------------------

PARAM_SETS = [
    # (S,    K,    T,    r,     sigma)
    (100.0, 100.0, 1.0,  0.05,  0.20),   # ATM standard
    (110.0,  90.0, 0.5,  0.03,  0.15),   # ITM call / OTM put, short tenor
    ( 80.0, 100.0, 2.0,  0.01,  0.30),   # OTM call / ITM put, low rate, high vol
    (100.0, 100.0, 0.1,  0.10,  0.05),   # near-expiry, high rate, low vol
    ( 50.0,  55.0, 1.5,  0.04,  0.40),   # OTM call, high vol
]


# ---------------------------------------------------------------------------
# 1. Put-call parity
# ---------------------------------------------------------------------------

class TestPutCallParity:
    @pytest.mark.parametrize("S, K, T, r, sigma", PARAM_SETS)
    def test_put_call_parity(self, S, K, T, r, sigma):
        """C - P must equal S - K*exp(-rT) to machine precision."""
        C = black_scholes(S, K, T, r, sigma, "call")
        P = black_scholes(S, K, T, r, sigma, "put")
        lhs = C - P
        rhs = S - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10, (
            f"Parity violated for S={S}, K={K}, T={T}: lhs={lhs:.8f}, rhs={rhs:.8f}"
        )


# ---------------------------------------------------------------------------
# 2. Monte Carlo convergence to Black-Scholes
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_converges_to_black_scholes(self):
        """500k sims should land within $0.05 of BS for both call and put."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        bs_call = black_scholes(S, K, T, r, sigma, "call")
        bs_put  = black_scholes(S, K, T, r, sigma, "put")

        mc_call, _ = monte_carlo_price(S, K, T, r, sigma, "call", 500_000, seed=0)
        mc_put,  _ = monte_carlo_price(S, K, T, r, sigma, "put",  500_000, seed=0)

        assert abs(mc_call - bs_call) < 0.05, (
            f"MC call {mc_call:.4f} too far from BS {bs_call:.4f}"
        )
        assert abs(mc_put - bs_put) < 0.05, (
            f"MC put {mc_put:.4f} too far from BS {bs_put:.4f}"
        )

    def test_seed_reproducibility(self):
        """Same seed must produce identical prices across two calls."""
        args = (100.0, 100.0, 1.0, 0.05, 0.20, "call", 10_000)
        price_a, _ = monte_carlo_price(*args, seed=99)
        price_b, _ = monte_carlo_price(*args, seed=99)
        assert price_a == price_b

    def test_returns_positive_std_error(self):
        """Standard error must be strictly positive for any finite simulation."""
        price, se = monte_carlo_price(100.0, 100.0, 1.0, 0.05, 0.20, "call", 1_000, seed=1)
        assert se > 0.0


# ---------------------------------------------------------------------------
# 3. Binomial tree convergence to Black-Scholes
# ---------------------------------------------------------------------------

class TestBinomialTree:
    def test_converges_to_black_scholes(self):
        """1000-step CRR tree must land within $0.01 of BS for call and put."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        bs_call = black_scholes(S, K, T, r, sigma, "call")
        bs_put  = black_scholes(S, K, T, r, sigma, "put")

        bt_call = binomial_tree_price(S, K, T, r, sigma, "call", 1000)
        bt_put  = binomial_tree_price(S, K, T, r, sigma, "put",  1000)

        assert abs(bt_call - bs_call) < 0.01, (
            f"BT call {bt_call:.6f} too far from BS {bs_call:.6f}"
        )
        assert abs(bt_put - bs_put) < 0.01, (
            f"BT put {bt_put:.6f} too far from BS {bs_put:.6f}"
        )

    def test_single_step_replicates_formula(self):
        """1-step tree must satisfy exact one-period replication pricing."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        dt = T
        u  = np.exp(sigma * np.sqrt(dt))
        d  = 1.0 / u
        p  = (np.exp(r * dt) - d) / (u - d)
        expected_call = np.exp(-r * T) * (p * max(S * u - K, 0) + (1 - p) * max(S * d - K, 0))
        bt_call = binomial_tree_price(S, K, T, r, sigma, "call", 1)
        assert abs(bt_call - expected_call) < 1e-12


# ---------------------------------------------------------------------------
# 4. Intrinsic value at expiry (T=0)
# ---------------------------------------------------------------------------

class TestIntrinsicValueAtExpiry:
    @pytest.mark.parametrize("S, K, expected_call, expected_put", [
        (110.0, 100.0, 10.0, 0.0),   # ITM call, OTM put
        ( 90.0, 100.0,  0.0, 10.0),  # OTM call, ITM put
        (100.0, 100.0,  0.0, 0.0),   # ATM: both worth zero
    ])
    def test_t0_returns_intrinsic(self, S, K, expected_call, expected_put):
        """At T=0, BS must return max(S-K,0) for calls, max(K-S,0) for puts."""
        assert black_scholes(S, K, 0.0, 0.05, 0.20, "call") == pytest.approx(expected_call)
        assert black_scholes(S, K, 0.0, 0.05, 0.20, "put")  == pytest.approx(expected_put)


# ---------------------------------------------------------------------------
# 5. Zero volatility (sigma→0)
# ---------------------------------------------------------------------------

class TestZeroVolatility:
    def test_call_sigma_zero(self):
        """With sigma=0, call = max(S - K*exp(-rT), 0): deterministic forward payoff."""
        S, K, T, r, sigma = 100.0, 95.0, 1.0, 0.05, 1e-10
        expected = max(S - K * np.exp(-r * T), 0.0)
        result = black_scholes(S, K, T, r, sigma, "call")
        assert abs(result - expected) < 1e-4

    def test_put_sigma_zero_otm(self):
        """With sigma=0, OTM put (S > K*exp(-rT)) should be worth ~0."""
        S, K, T, r, sigma = 100.0, 90.0, 1.0, 0.05, 1e-10
        result = black_scholes(S, K, T, r, sigma, "put")
        assert result < 1e-4

    def test_sigma_exactly_zero_edge_case(self):
        """sigma=0 edge branch: ITM call returns S - K*exp(-rT)."""
        S, K, T, r = 110.0, 100.0, 1.0, 0.05
        expected_call = max(S - K * np.exp(-r * T), 0.0)
        result = black_scholes(S, K, T, r, 0.0, "call")
        assert abs(result - expected_call) < 1e-10


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    @pytest.mark.parametrize("bad_kwargs, match", [
        ({"S": -1.0},          "S must be positive"),
        ({"S":  0.0},          "S must be positive"),
        ({"K": -5.0},          "K must be positive"),
        ({"K":  0.0},          "K must be positive"),
        ({"T": -0.1},          "T must be non-negative"),
        ({"sigma": -0.2},      "sigma must be positive"),
        ({"option_type": "forward"}, "option_type must be"),
    ])
    def test_black_scholes_raises(self, bad_kwargs, match):
        base = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20, "option_type": "call"}
        params = {**base, **bad_kwargs}
        if match is None:
            black_scholes(**params)   # should NOT raise
        else:
            with pytest.raises(ValueError, match=match):
                black_scholes(**params)

    def test_monte_carlo_negative_S_raises(self):
        with pytest.raises(ValueError, match="S must be positive"):
            monte_carlo_price(-1.0, 100.0, 1.0, 0.05, 0.2, "call", 1000)

    def test_binomial_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            binomial_tree_price(100.0, 100.0, 1.0, 0.05, -0.2, "call", 10)

    def test_binomial_zero_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps must be at least 1"):
            binomial_tree_price(100.0, 100.0, 1.0, 0.05, 0.2, "call", 0)


# ---------------------------------------------------------------------------
# 7. Vectorised inputs
# ---------------------------------------------------------------------------

class TestVectorizedInputs:
    def test_array_S_returns_same_shape(self):
        """Passing an array of spot prices returns an array of the same shape."""
        S_arr = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        prices = black_scholes(S_arr, 100.0, 1.0, 0.05, 0.20, "call")
        assert prices.shape == S_arr.shape

    def test_array_values_match_scalar_loop(self):
        """Each element of the vectorised result must equal the scalar call."""
        S_arr = np.array([80.0, 100.0, 120.0])
        prices = black_scholes(S_arr, 100.0, 1.0, 0.05, 0.20, "put")
        for i, S in enumerate(S_arr):
            expected = black_scholes(S, 100.0, 1.0, 0.05, 0.20, "put")
            assert prices[i] == pytest.approx(expected, rel=1e-10)

    def test_array_sigma_returns_same_shape(self):
        """Passing an array of vols returns an array of the same shape."""
        sigma_arr = np.linspace(0.10, 0.50, 8)
        prices = black_scholes(100.0, 100.0, 1.0, 0.05, sigma_arr, "call")
        assert prices.shape == sigma_arr.shape

    def test_monotone_call_in_S(self):
        """Call price must be strictly increasing in S (all else equal)."""
        S_arr = np.linspace(80.0, 120.0, 20)
        prices = black_scholes(S_arr, 100.0, 1.0, 0.05, 0.20, "call")
        assert np.all(np.diff(prices) > 0)

    def test_monotone_put_in_S(self):
        """Put price must be strictly decreasing in S (all else equal)."""
        S_arr = np.linspace(80.0, 120.0, 20)
        prices = black_scholes(S_arr, 100.0, 1.0, 0.05, 0.20, "put")
        assert np.all(np.diff(prices) < 0)
