import math
import re
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import simulation_functions as sim
except ModuleNotFoundError:  # allow running from repo root
    from Simulation import simulation_functions as sim



lottery = {10: 0.8, 5:0.2}#{72: 0.25, 20: 0.25, -80: 0.25, -40: 0.25}
second_lottery =None  # {10: 0.8, 5:0.2}

observed_ce = None
synthetic_n = 100
seed = 123
r_true = 0.0
r = 0.97
alpha = 0.88
lambda_ = 2.25
gamma = 0.61
sigma = 5.0
r_min = None
r_max = None
points = 400


def _lottery_dict_to_arrays(lottery_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a lottery dict to arrays of outcomes and probabilities."""
    if not isinstance(lottery_dict, dict) or not lottery_dict:
        raise ValueError("Lottery must be a non-empty dict.")

    try:
        keys = np.array([float(k) for k in lottery_dict.keys()], dtype=float)
        values = np.array([float(v) for v in lottery_dict.values()], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Lottery keys and values must be numeric.") from exc

    def looks_like_probs(arr: np.ndarray) -> bool:
        return np.all(arr >= 0) and np.all(arr <= 1) and abs(arr.sum() - 1.0) <= 1e-6

    keys_are_probs = looks_like_probs(keys)
    values_are_probs = looks_like_probs(values)
    if keys_are_probs and not values_are_probs:
        probs = keys
        outcomes = values
    else:
        outcomes = keys
        probs = values

    merged = {}
    for outcome, prob in zip(outcomes, probs):
        if not np.isfinite(outcome) or not np.isfinite(prob):
            raise ValueError("Lottery values/probabilities must be finite.")
        if prob < 0:
            raise ValueError("Lottery probabilities must be >= 0.")
        merged[outcome] = merged.get(outcome, 0.0) + prob

    outcomes = np.array(list(merged.keys()), dtype=float)
    probs = np.array([merged[v] for v in merged.keys()], dtype=float)
    total_prob = probs.sum()
    if total_prob <= 0:
        raise ValueError("Lottery probabilities must sum to a positive value.")
    probs = probs / total_prob
    return outcomes, probs


def sum_lotteries(lottery_a: dict, lottery_b: dict) -> dict:
    """Return the sum distribution of two lotteries (outcome->prob)."""
    outcomes_a, probs_a = _lottery_dict_to_arrays(lottery_a)
    outcomes_b, probs_b = _lottery_dict_to_arrays(lottery_b)
    merged = {}
    for value_a, prob_a in zip(outcomes_a, probs_a):
        for value_b, prob_b in zip(outcomes_b, probs_b):
            total = value_a + value_b
            merged[total] = merged.get(total, 0.0) + prob_a * prob_b
    return merged


def _parse_observed_ce(raw: str) -> np.ndarray:
    """Parse comma/space-separated observed CE values."""
    if raw is None:
        raise ValueError("Observed CE string cannot be empty.")
    parts = re.split(r"[,\s;]+", raw.strip())
    values = [float(p) for p in parts if p]
    if not values:
        raise ValueError("No observed CE values provided.")
    return np.array(values, dtype=float)


def _compute_ce(Z: np.ndarray, p: np.ndarray, R: float, alpha: float, lambda_: float, gamma: float, r: float) -> float:
    """Compute CE using the simulation_reference_point / range_simulation form."""
    if Z.size == 0:
        return float("nan")
    _, ce = sim.ce(alpha, lambda_, gamma, r, Z, p, R)
    return float(ce)


def _logpdf_normal(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Normal log-pdf evaluated elementwise."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    z = (x - mean) / sigma
    return -0.5 * z * z - math.log(sigma * math.sqrt(2.0 * math.pi))


def _synthetic_observations_model(
    Z: np.ndarray,
    p: np.ndarray,
    n: int,
    R_true: float,
    alpha: float,
    lambda_: float,
    gamma: float,
    r: float,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Generate synthetic observations: CE(R_true) + eps."""
    if n <= 0:
        raise ValueError("synthetic-n must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive for synthetic data.")
    rng = np.random.default_rng(seed)
    ce0 = _compute_ce(Z, p, R_true, alpha, lambda_, gamma, r)
    eps = rng.normal(0.0, sigma, size=n)
    return ce0 + eps


def _compute_loglikelihood_series(
    observed: np.ndarray,
    Z: np.ndarray,
    p: np.ndarray,
    alpha: float,
    lambda_: float,
    gamma: float,
    r: float,
    sigma: float,
    r_min: float,
    r_max: float,
    points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute log-likelihood over grid of R for a simple lottery."""
    if observed.size == 0:
        raise ValueError("No observations provided.")
    if points <= 1:
        raise ValueError("points must be > 1.")

    if r_min is None:
        r_min = float(np.min(Z)) * 1.5
    if r_max is None:
        r_max = float(np.max(Z)) * 1.5
    if r_min >= r_max:
        raise ValueError("r-min must be less than r-max.")

    R_values = np.linspace(r_min, r_max, points)
    total_loglik = np.zeros_like(R_values, dtype=float)

    for idx, R in enumerate(R_values):
        ce0 = _compute_ce(Z, p, float(R), alpha, lambda_, gamma, r)
        total_loglik[idx] = float(np.sum(_logpdf_normal(observed, ce0, sigma)))

    return R_values, total_loglik


def plot_loglikelihood(
    observed: np.ndarray,
    Z: np.ndarray,
    p: np.ndarray,
    alpha: float,
    lambda_: float,
    gamma: float,
    r: float,
    sigma: float,
    r_min: float,
    r_max: float,
    points: int,
) -> None:
    """Plot log-likelihood curve over R for a simple lottery."""
    R_values, total_loglik = _compute_loglikelihood_series(
        observed=observed,
        Z=Z,
        p=p,
        alpha=alpha,
        lambda_=lambda_,
        gamma=gamma,
        r=r,
        sigma=sigma,
        r_min=r_min,
        r_max=r_max,
        points=points,
    )

    finite_mask = np.isfinite(total_loglik)
    if np.any(finite_mask):
        max_idx = int(np.nanargmax(total_loglik))
        max_r = float(R_values[max_idx])
        max_ll = float(total_loglik[max_idx])
        print(f"Max log-likelihood: {max_ll:.4f} at R={max_r:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(R_values, total_loglik, color="black", linewidth=2.0)
    plt.xlabel("Reference Point R")
    plt.ylabel("Log-likelihood")
    plt.title("Log-likelihood vs Reference Point (simple lottery)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run with parameters defined at the top of this file."""
    lottery_to_use = lottery
    if second_lottery is not None:
        lottery_to_use = sum_lotteries(lottery, second_lottery)
    Z, p = _lottery_dict_to_arrays(lottery_to_use)

    if observed_ce is not None:
        if isinstance(observed_ce, (list, tuple, np.ndarray)):
            observed = np.asarray(observed_ce, dtype=float)
        else:
            observed = _parse_observed_ce(str(observed_ce))
    else:
        observed = _synthetic_observations_model(
            Z=Z,
            p=p,
            n=synthetic_n,
            R_true=r_true,
            alpha=alpha,
            lambda_=lambda_,
            gamma=gamma,
            r=r,
            sigma=sigma,
            seed=seed,
        )

    plot_loglikelihood(
        observed=observed,
        Z=Z,
        p=p,
        alpha=alpha,
        lambda_=lambda_,
        gamma=gamma,
        r=r,
        sigma=sigma,
        r_min=r_min,
        r_max=r_max,
        points=points,
    )


if __name__ == "__main__":
    main()
