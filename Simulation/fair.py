from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===============================
# 1. Model primitives
# ===============================

# Set your parameters here
alpha = 0.6   # social weight on dictator
beta  = 0.4   # social weight on recipient
mu    = 5.0   # mean of risky pie
rho   = 0.02  # CARA coefficient


@dataclass(frozen=True)
class DictatorParams:
    """
    Container for the preference and technology parameters of the risky dictator game.
    """
    alpha: float
    beta: float
    mu: float
    rho: float
    name: Optional[str] = None

    def label(self) -> str:
        """
        User-friendly label that is used in tables/plots.
        """
        return self.name or (
            f"alpha={self.alpha}, beta={self.beta}, mu={self.mu}, rho={self.rho}"
        )


def _resolve_params(params: Optional[DictatorParams],
                    alpha_fallback: float,
                    beta_fallback: float,
                    mu_fallback: float,
                    rho_fallback: float) -> Tuple[float, float, float, float]:
    """
    Helper that converts an optional DictatorParams instance into primitives.
    """
    if params is None:
        return alpha_fallback, beta_fallback, mu_fallback, rho_fallback
    return params.alpha, params.beta, params.mu, params.rho

def FOC_r(r, s, alpha=alpha, beta=beta, mu=mu, rho=rho):
    """
    First order condition dV/dr = 0 for a given r and s.
    Returns F(r,s) = dV/dr.
    """
    A = -rho * r * mu + 0.5 * rho**2 * r**2 * s**2
    B = -rho * (1.0 - r) * mu + 0.5 * rho**2 * (1.0 - r)**2 * s**2

    term1 = alpha * np.exp(A) * (-mu + rho * r * s**2)
    term2 = beta  * np.exp(B) * (mu - rho * (1.0 - r) * s**2)
    return term1 + term2


# ===============================
# 2. Generic bisection solver
# ===============================

def bisect_root(func, a, b, tol=1e-8, max_iter=100):
    """
    Simple bisection method to find root of func(x) in [a,b].
    Requires func(a) and func(b) to have opposite signs.
    """
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        raise ValueError("Bisection failed: func(a) and func(b) have same sign.")

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = func(m)

        if abs(fm) < tol or (b - a) / 2 < tol:
            return m

        # keep the subinterval where the sign changes
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return 0.5 * (a + b)


# ===============================
# 3. r*(s): optimal sharing rule
# ===============================

def r_star_given_s(s,
                   alpha=alpha,
                   beta=beta,
                   mu=mu,
                   rho=rho,
                   params: Optional[DictatorParams] = None):
    """
    For a given risk level s, solve FOC_r(r,s) = 0 for r in [0,1].
    Falls back to the boundary if the derivative keeps the same sign.
    """
    alpha, beta, mu, rho = _resolve_params(params, alpha, beta, mu, rho)
    func = lambda r: FOC_r(r, s, alpha=alpha, beta=beta, mu=mu, rho=rho)

    lower, upper = 0.0, 1.0
    f_lower = func(lower)
    if abs(f_lower) < 1e-10:
        return lower

    f_upper = func(upper)
    if abs(f_upper) < 1e-10:
        return upper

    if f_lower * f_upper > 0:
        # With CARA preferences V(r;s) is concave, so opposite signs fail only at a boundary optimum.
        return lower if f_lower < 0 else upper

    return bisect_root(func, lower, upper)


# ===============================
# 4. Inverse: s*(r) calibration
# ===============================

def s_star_given_r(r_target, s_low=0.0, s_high=20,
                   alpha=alpha, beta=beta, mu=mu, rho=rho,
                   params: Optional[DictatorParams] = None):
    """
    For a given sharing rule r_target, find s such that FOC_r(r_target,s) = 0.
    This 'calibrates' the risk level s that rationalizes the observed r_target.
    """
    alpha, beta, mu, rho = _resolve_params(params, alpha, beta, mu, rho)
    func = lambda s: FOC_r(r_target, s, alpha=alpha, beta=beta, mu=mu, rho=rho)
    return bisect_root(func, s_low, s_high)


# ===============================
# 5. Simulation helpers
# ===============================

def simulate_r_s_grid(s_values: Sequence[float],
                      param_sets: Iterable[DictatorParams]) -> pd.DataFrame:
    """
    Evaluate r*(s) for a grid of s-values across several parameterizations.
    """
    records = []
    s_values = np.asarray(s_values, dtype=float)
    for params in param_sets:
        for s in s_values:
            r_star = r_star_given_s(s, params=params)
            records.append({
                "s": s,
                "r_star": r_star,
                "alpha": params.alpha,
                "beta": params.beta,
                "mu": params.mu,
                "rho": params.rho,
                "param_set": params.label(),
            })
    return pd.DataFrame.from_records(records)


def plot_r_vs_s(results_df: pd.DataFrame,
                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Overlay r*(s) curves for all parameter sets contained in the supplied DataFrame.
    """
    if ax is None:
        _, ax = plt.subplots()

    for label, group in results_df.groupby("param_set"):
        grp = group.sort_values("s")
        ax.plot(grp["s"], grp["r_star"], label=label)

    ax.set_xlabel("s (standard deviation of risky pie)")
    ax.set_ylabel("r*(s) (optimal share to dictator)")
    ax.set_title("Optimal sharing rule across parameterizations")
    ax.grid(True)
    ax.legend()
    return ax


# ===============================
# 6. Example: calibrate r*(s) on a grid
# ===============================

if __name__ == "__main__":
    # Grid of risk levels
    s_grid = np.linspace(0.0, 50, 100)

    # Parameterizations to explore
    parameter_sets = [
        
        # Mildly risk-averse (close to experimental estimates)
        DictatorParams(0.7, 0.3, 5, 0.2, name="rho = 0.2, mu = 5"),
        DictatorParams(0.6, 0.4, 5, 0.17, name="rho = 0.17, mu = 5"),
        DictatorParams(0.6, 0.4, 5, 0.15, name="rho = 0.15, mu = 5"),
        DictatorParams(0.6, 0.4, 10, 0.2, name="rho = 0.2, mu = 10"),
        DictatorParams(0.6, 0.4, 10, 0.17, name="rho = 0.17, mu = 10"),
        DictatorParams(0.6, 0.4, 10, 0.15, name="rho = 0.15, mu = 10"),


        DictatorParams(0.6, 0.4, -5, 0.2, name="rho = 0.2, mu = -5"),
        DictatorParams(0.6, 0.4, -5, 0.17, name="rho = 0.17, mu = -5"),
        DictatorParams(0.6, 0.4, -5, 0.15, name="rho = 0.15, mu = -5"),
        DictatorParams(0.6, 0.4, -10, 0.2, name="rho = 0.2, mu = -10"),
        DictatorParams(0.6, 0.4, -10, 0.17, name="rho = 0.17, mu = -10"),
        DictatorParams(0.6, 0.4, -10, 0.15, name="rho = 0.15, mu = -10"),
    ]

    # Run the simulation grid and inspect the first rows
    simulation_results = simulate_r_s_grid(s_grid, parameter_sets)
    print(simulation_results.head())

    # Visualize r*(s) separately for positive and negative mean pies
    pos_results = simulation_results[simulation_results["mu"] >= 0]
    if not pos_results.empty:
        ax_pos = plot_r_vs_s(pos_results)
        #ax_pos.set_title("Optimal sharing rule (mu >= 0)")
        fig_pos = ax_pos.figure
        fig_pos.savefig("r_vs_s_mu_positive.eps", format="eps", bbox_inches="tight")
        plt.show()

    neg_results = simulation_results[simulation_results["mu"] < 0]
    if not neg_results.empty:
        ax_neg = plot_r_vs_s(neg_results)
        #ax_neg.set_title("Optimal sharing rule (mu < 0)")
        fig_neg = ax_neg.figure
        fig_neg.savefig("r_vs_s_mu_negative.eps", format="eps", bbox_inches="tight")
        plt.show()
