
# ============================================================
# Robust Gibbs-Thompson Sampling (RG-TS) Simulation Suite
# Colab-ready Python script for heavy-tailed bandit verification
# ------------------------------------------------------------
# Implements the paper's generalized Gibbs posterior idea using
# Catoni's psi influence function and ULA sampling, and compares
# RG-TS against practical baselines under skewed/heavy-tailed
# reward scenarios.
# ============================================================

import os
import math
import json
import shutil
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------
# Configuration
# -----------------------------
RUN_PROFILE = "fast"   # "fast" or "paper"
SEED = 20260330
SAVE_ROOT = "rgts_sim_outputs"

PROFILE = {
    "fast": {
        "T": 500,
        "n_rep": 30,
        "rgts_M": 18,
        "posterior_grid_points": 220,
        "distribution_draws": 2500,
    },
    "paper": {
        "T": 1200,
        "n_rep": 120,
        "rgts_M": 30,
        "posterior_grid_points": 320,
        "distribution_draws": 5000,
    }
}[RUN_PROFILE]

GLOBAL_CONFIG = {
    "seed": SEED,
    "T": PROFILE["T"],
    "n_rep": PROFILE["n_rep"],
    "warm_start_each_arm_once": True,
    "algorithms": ["RGTS", "GaussianTS", "CatoniUCB", "UCB1"],
    # prior / posterior settings
    "prior_mean": 0.0,
    "prior_sd": 2.0,
    "gaussian_ts_obs_var": 1.0,   # deliberately misspecified under heavy tails
    # RGTS settings
    "rgts_lambda": 1.0,
    "rgts_eta": 0.02,
    "rgts_M": PROFILE["rgts_M"],
    "alpha_coef": 0.70,           # adaptive alpha coefficient
    "alpha_min": 0.03,
    "alpha_max": 1.25,
    # UCB settings
    "ucb_scale": 1.0,
    "catoni_ucb_scale": 1.0,
    # plotting / saving
    "posterior_grid_points": PROFILE["posterior_grid_points"],
    "distribution_draws": PROFILE["distribution_draws"],
}

# -----------------------------
# Display helpers
# -----------------------------
def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

def show_df(df: pd.DataFrame, title: str = None, n: int = None):
    if title:
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))
    out = df if n is None else df.head(n)
    try:
        from IPython.display import display
        display(out)
    except Exception:
        print(out.to_string(index=False))

def ensure_dirs(root: str):
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "tables"), exist_ok=True)
    os.makedirs(os.path.join(root, "figures"), exist_ok=True)
    os.makedirs(os.path.join(root, "metadata"), exist_ok=True)

# -----------------------------
# Robust utilities
# -----------------------------
def catoni_psi(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = np.log1p(x[pos] + 0.5 * x[pos] ** 2)
    neg = ~pos
    out[neg] = -np.log1p(-x[neg] + 0.5 * x[neg] ** 2)
    return out

def catoni_psi_prime(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = (1.0 + x[pos]) / (1.0 + x[pos] + 0.5 * x[pos] ** 2)
    neg = ~pos
    out[neg] = (1.0 - x[neg]) / (1.0 - x[neg] + 0.5 * x[neg] ** 2)
    return out

def adaptive_alpha(n, eps, u_proxy, cfg):
    # The paper's theorem sketch suggests alpha proportional to (n/u)^(1/(1+eps)).
    # For numerical stability in finite simulation, we cap alpha to a practical range.
    raw = cfg["alpha_coef"] * ((max(n, 1) / max(u_proxy, 1e-8)) ** (1.0 / (1.0 + eps)))
    return float(np.clip(raw, cfg["alpha_min"], cfg["alpha_max"]))

def catoni_estimate(samples, alpha, init=None, max_iter=50, tol=1e-8):
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return 0.0
    if x.size == 1:
        return float(x[0])

    theta = np.median(x) if init is None else float(init)
    # Newton steps with damping
    for _ in range(max_iter):
        z = alpha * (x - theta)
        g = np.sum(catoni_psi(z))
        gp = -alpha * np.sum(catoni_psi_prime(z))
        step = g / (gp if abs(gp) > 1e-10 else np.sign(gp) * 1e-10 + 1e-10)
        step = np.clip(step, -5.0, 5.0)
        theta_new = theta - step
        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    return float(theta)

def robust_gibbs_gradient(mu, samples, prior_mean, prior_sd, lam, alpha):
    x = np.asarray(samples, dtype=float)
    prior_grad = (mu - prior_mean) / (prior_sd ** 2)
    if x.size == 0:
        return float(prior_grad)
    robust_term = -lam * alpha * np.sum(catoni_psi(alpha * (x - mu)))
    return float(prior_grad + robust_term)

def rgts_draw(samples, state, eps, u_proxy, rng, cfg):
    if len(samples) == 0:
        return float(rng.normal(cfg["prior_mean"], cfg["prior_sd"]))

    alpha = adaptive_alpha(len(samples), eps, u_proxy, cfg)
    init = state.get("last_catoni", None)
    theta = catoni_estimate(samples, alpha=alpha, init=init)
    state["last_catoni"] = theta

    for _ in range(cfg["rgts_M"]):
        grad = robust_gibbs_gradient(
            mu=theta,
            samples=samples,
            prior_mean=cfg["prior_mean"],
            prior_sd=cfg["prior_sd"],
            lam=cfg["rgts_lambda"],
            alpha=alpha,
        )
        theta = theta - cfg["rgts_eta"] * grad + math.sqrt(2.0 * cfg["rgts_eta"]) * rng.normal()

    return float(theta)

def gaussian_ts_draw(samples, rng, cfg):
    n = len(samples)
    m0 = cfg["prior_mean"]
    s0_2 = cfg["prior_sd"] ** 2
    sigma2 = cfg["gaussian_ts_obs_var"]
    if n == 0:
        post_mean = m0
        post_var = s0_2
    else:
        post_var = 1.0 / (1.0 / s0_2 + n / sigma2)
        post_mean = post_var * (m0 / s0_2 + np.sum(samples) / sigma2)
    return float(rng.normal(post_mean, math.sqrt(post_var)))

def gaussian_ts_post_mean(samples, cfg):
    n = len(samples)
    m0 = cfg["prior_mean"]
    s0_2 = cfg["prior_sd"] ** 2
    sigma2 = cfg["gaussian_ts_obs_var"]
    if n == 0:
        return m0
    post_var = 1.0 / (1.0 / s0_2 + n / sigma2)
    post_mean = post_var * (m0 / s0_2 + np.sum(samples) / sigma2)
    return float(post_mean)

def ucb1_index(samples, t, cfg):
    n = len(samples)
    if n == 0:
        return float("inf")
    return float(np.mean(samples) + cfg["ucb_scale"] * np.sqrt(2.0 * np.log(max(t, 2)) / n))

def catoni_ucb_index(samples, t, eps, u_proxy, cfg, state=None):
    n = len(samples)
    if n == 0:
        return float("inf")
    alpha = adaptive_alpha(n, eps, u_proxy, cfg)
    init = None if state is None else state.get("last_catoni", None)
    est = catoni_estimate(samples, alpha=alpha, init=init)
    if state is not None:
        state["last_catoni"] = est
    bonus = cfg["catoni_ucb_scale"] * np.sqrt(2.0 * np.log(max(t, 2)) / n)
    return float(est + bonus)

# -----------------------------
# Reward scenarios
# -----------------------------
@dataclass
class BanditScenario:
    name: str
    description: str
    means: List[float]
    eps: float
    u_proxy: float
    draw_fn: Callable[[int, np.random.Generator], float]
    arm_labels: List[str]

def pareto_arm(mean_target, shape, rng):
    xm = mean_target * (shape - 1.0) / shape
    return float(xm * (1.0 + rng.pareto(shape)))

def lognormal_arm(mean_target, sigma, rng):
    mu_log = np.log(mean_target) - 0.5 * sigma ** 2
    return float(rng.lognormal(mean=mu_log, sigma=sigma))

def gamma_arm(mean_target, shape, rng):
    scale = mean_target / shape
    return float(rng.gamma(shape=shape, scale=scale))

def make_scenarios():
    scenarios = []

    # Scenario 1: Infinite-variance Pareto arms (shape < 2) with finite mean
    means1 = [1.00, 0.92, 0.84]
    shape1 = 1.45
    def draw1(arm, rng):
        return pareto_arm(means1[arm], shape1, rng)
    scenarios.append(BanditScenario(
        name="Pareto_InfiniteVariance",
        description="All arms are Pareto with finite mean and infinite variance (shape 1.45).",
        means=means1,
        eps=0.35,
        u_proxy=8.0,
        draw_fn=draw1,
        arm_labels=[f"Arm {i+1}" for i in range(len(means1))]
    ))

    # Scenario 2: Rare whale contamination on a suboptimal arm
    # Optimal arm is stable; arm 2 occasionally emits giant rewards but remains suboptimal in mean.
    means2 = [1.00, 0.95, 0.86]
    whale_prob = [0.0, 0.006, 0.0]
    whale_size = [0.0, 60.0, 0.0]
    base_means = [1.00, means2[1] - whale_prob[1] * whale_size[1], 0.86]
    base_shapes = [2.5, 2.0, 2.0]
    def draw2(arm, rng):
        if rng.uniform() < whale_prob[arm]:
            return float(whale_size[arm] + rng.gamma(shape=2.0, scale=0.5))
        return gamma_arm(base_means[arm], base_shapes[arm], rng)
    scenarios.append(BanditScenario(
        name="Whale_Contamination",
        description="A suboptimal arm produces rare giant outliers ('whales') that can mislead naive TS.",
        means=means2,
        eps=0.50,
        u_proxy=18.0,
        draw_fn=draw2,
        arm_labels=[f"Arm {i+1}" for i in range(len(means2))]
    ))

    # Scenario 3: Strongly skewed log-normal rewards with finite variance
    means3 = [1.00, 0.93, 0.87]
    sigma3 = 1.2
    def draw3(arm, rng):
        return lognormal_arm(means3[arm], sigma3, rng)
    scenarios.append(BanditScenario(
        name="Lognormal_Skew",
        description="All arms are highly skewed log-normal rewards with finite variance.",
        means=means3,
        eps=0.80,
        u_proxy=6.0,
        draw_fn=draw3,
        arm_labels=[f"Arm {i+1}" for i in range(len(means3))]
    ))

    return scenarios

# -----------------------------
# Simulation engine
# -----------------------------
def final_point_estimate(algorithm, histories, scenario, cfg, arm_states):
    est = []
    for k, hist in enumerate(histories):
        if algorithm == "RGTS":
            alpha = adaptive_alpha(len(hist), scenario.eps, scenario.u_proxy, cfg)
            est.append(catoni_estimate(hist, alpha=alpha, init=arm_states[k].get("last_catoni", None)) if len(hist) else cfg["prior_mean"])
        elif algorithm == "GaussianTS":
            est.append(gaussian_ts_post_mean(hist, cfg))
        elif algorithm == "CatoniUCB":
            if len(hist) == 0:
                est.append(cfg["prior_mean"])
            else:
                alpha = adaptive_alpha(len(hist), scenario.eps, scenario.u_proxy, cfg)
                est.append(catoni_estimate(hist, alpha=alpha, init=arm_states[k].get("last_catoni", None)))
        else:
            est.append(np.mean(hist) if len(hist) else cfg["prior_mean"])
    return np.array(est, dtype=float)

def run_one_algorithm(algorithm, scenario: BanditScenario, T: int, rng, cfg):
    K = len(scenario.means)
    opt_arm = int(np.argmax(scenario.means))
    opt_mean = float(np.max(scenario.means))

    histories = [[] for _ in range(K)]
    arm_states = [dict() for _ in range(K)]
    cum_regret = np.zeros(T)
    opt_pull = np.zeros(T)
    chosen_arms = np.zeros(T, dtype=int)
    rewards = np.zeros(T)

    total_regret = 0.0

    for t in range(T):
        if cfg["warm_start_each_arm_once"] and t < K:
            arm = t
        else:
            if algorithm == "RGTS":
                sampled_means = [
                    rgts_draw(histories[k], arm_states[k], scenario.eps, scenario.u_proxy, rng, cfg)
                    for k in range(K)
                ]
                arm = int(np.argmax(sampled_means))
            elif algorithm == "GaussianTS":
                sampled_means = [gaussian_ts_draw(histories[k], rng, cfg) for k in range(K)]
                arm = int(np.argmax(sampled_means))
            elif algorithm == "CatoniUCB":
                indices = [
                    catoni_ucb_index(histories[k], t + 1, scenario.eps, scenario.u_proxy, cfg, arm_states[k])
                    for k in range(K)
                ]
                arm = int(np.argmax(indices))
            elif algorithm == "UCB1":
                indices = [ucb1_index(histories[k], t + 1, cfg) for k in range(K)]
                arm = int(np.argmax(indices))
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        reward = scenario.draw_fn(arm, rng)
        histories[arm].append(reward)

        chosen_arms[t] = arm
        rewards[t] = reward
        total_regret += opt_mean - scenario.means[arm]
        cum_regret[t] = total_regret
        opt_pull[t] = 1.0 if arm == opt_arm else 0.0

    final_est = final_point_estimate(algorithm, histories, scenario, cfg, arm_states)
    best_arm_correct = int(np.argmax(final_est) == opt_arm)
    pull_counts = np.array([len(h) for h in histories], dtype=int)

    return {
        "cum_regret": cum_regret,
        "opt_pull": opt_pull,
        "chosen_arms": chosen_arms,
        "rewards": rewards,
        "pull_counts": pull_counts,
        "best_arm_correct": best_arm_correct,
        "final_reward": float(np.sum(rewards)),
        "final_regret": float(cum_regret[-1]),
        "final_estimates": final_est,
        "opt_arm_est_error": float(abs(final_est[opt_arm] - scenario.means[opt_arm])),
    }

def run_simulation_suite(cfg):
    rng_master = np.random.default_rng(cfg["seed"])
    scenarios = make_scenarios()
    algorithms = cfg["algorithms"]
    T = cfg["T"]
    n_rep = cfg["n_rep"]

    run_records = []
    trajectory_records = []

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario.name}")
        for algorithm in algorithms:
            regrets = np.zeros((n_rep, T))
            opts = np.zeros((n_rep, T))
            pull_mat = np.zeros((n_rep, len(scenario.means)), dtype=int)

            final_regrets = []
            final_rewards = []
            correct = []
            est_err = []

            for r in range(n_rep):
                rng = np.random.default_rng(rng_master.integers(0, 10**9))
                out = run_one_algorithm(algorithm, scenario, T, rng, cfg)

                regrets[r, :] = out["cum_regret"]
                opts[r, :] = out["opt_pull"]
                pull_mat[r, :] = out["pull_counts"]

                final_regrets.append(out["final_regret"])
                final_rewards.append(out["final_reward"])
                correct.append(out["best_arm_correct"])
                est_err.append(out["opt_arm_est_error"])

                run_records.append({
                    "scenario": scenario.name,
                    "algorithm": algorithm,
                    "rep": r + 1,
                    "final_regret": out["final_regret"],
                    "total_reward": out["final_reward"],
                    "best_arm_correct": out["best_arm_correct"],
                    "optimal_pull_share": np.mean(out["opt_pull"]),
                    "opt_arm_est_error": out["opt_arm_est_error"],
                    **{f"pull_share_arm_{k+1}": out["pull_counts"][k] / T for k in range(len(scenario.means))}
                })

            mean_regret = regrets.mean(axis=0)
            se_regret = regrets.std(axis=0, ddof=1) / np.sqrt(n_rep)
            mean_opt = opts.mean(axis=0)
            se_opt = opts.std(axis=0, ddof=1) / np.sqrt(n_rep)

            for t in range(T):
                trajectory_records.append({
                    "scenario": scenario.name,
                    "algorithm": algorithm,
                    "time": t + 1,
                    "mean_regret": mean_regret[t],
                    "se_regret": se_regret[t],
                    "mean_optimal_pull": mean_opt[t],
                    "se_optimal_pull": se_opt[t],
                })

    return (
        pd.DataFrame(run_records),
        pd.DataFrame(trajectory_records),
        scenarios,
    )

# -----------------------------
# Posterior sensitivity analysis
# -----------------------------
def robust_loss_scalar(mu, x, alpha):
    d = x - mu
    if abs(d) < 1e-12:
        return 0.0
    integrand = lambda u: float(catoni_psi(np.array([alpha * u]))[0])
    if d > 0:
        val, _ = quad(integrand, 0.0, d, limit=100)
        return float(val)
    else:
        val, _ = quad(integrand, d, 0.0, limit=100)
        return float(-val)

def rgts_grid_density(grid, samples, cfg, eps=0.5, u_proxy=10.0):
    alpha = adaptive_alpha(len(samples), eps, u_proxy, cfg)
    m0 = cfg["prior_mean"]
    s0 = cfg["prior_sd"]
    lam = cfg["rgts_lambda"]

    logdens = []
    for mu in grid:
        prior_log = norm.logpdf(mu, loc=m0, scale=s0)
        loss = sum(robust_loss_scalar(mu, x, alpha) for x in samples)
        logdens.append(prior_log - lam * loss)
    logdens = np.array(logdens)
    logdens -= np.max(logdens)
    dens = np.exp(logdens)
    dens /= np.trapz(dens, grid)
    return dens

def gaussian_ts_grid_density(grid, samples, cfg):
    n = len(samples)
    m0 = cfg["prior_mean"]
    s0_2 = cfg["prior_sd"] ** 2
    sigma2 = cfg["gaussian_ts_obs_var"]
    if n == 0:
        post_mean = m0
        post_var = s0_2
    else:
        post_var = 1.0 / (1.0 / s0_2 + n / sigma2)
        post_mean = post_var * (m0 / s0_2 + np.sum(samples) / sigma2)
    dens = norm.pdf(grid, loc=post_mean, scale=np.sqrt(post_var))
    dens /= np.trapz(dens, grid)
    return dens

def posterior_sensitivity_experiment(cfg):
    rng = np.random.default_rng(cfg["seed"] + 777)
    base_mean = 0.80
    base_shape = 2.0
    base = np.array([gamma_arm(base_mean, base_shape, rng) for _ in range(24)], dtype=float)
    whale = 80.0
    with_whale = np.append(base, whale)

    grid = np.linspace(-0.5, 8.0, cfg["posterior_grid_points"])

    dens_gauss_base = gaussian_ts_grid_density(grid, base, cfg)
    dens_gauss_whale = gaussian_ts_grid_density(grid, with_whale, cfg)
    dens_rgts_base = rgts_grid_density(grid, base, cfg, eps=0.5, u_proxy=18.0)
    dens_rgts_whale = rgts_grid_density(grid, with_whale, cfg, eps=0.5, u_proxy=18.0)

    def posterior_mean(grid, dens):
        return float(np.trapz(grid * dens, grid))

    table = pd.DataFrame({
        "model": ["GaussianTS", "GaussianTS", "RGTS", "RGTS"],
        "sample_case": ["Base", "Base + one whale", "Base", "Base + one whale"],
        "posterior_mean": [
            posterior_mean(grid, dens_gauss_base),
            posterior_mean(grid, dens_gauss_whale),
            posterior_mean(grid, dens_rgts_base),
            posterior_mean(grid, dens_rgts_whale),
        ],
        "posterior_mode": [
            float(grid[np.argmax(dens_gauss_base)]),
            float(grid[np.argmax(dens_gauss_whale)]),
            float(grid[np.argmax(dens_rgts_base)]),
            float(grid[np.argmax(dens_rgts_whale)]),
        ],
    })

    curves = {
        "grid": grid,
        "gauss_base": dens_gauss_base,
        "gauss_whale": dens_gauss_whale,
        "rgts_base": dens_rgts_base,
        "rgts_whale": dens_rgts_whale,
    }
    return table, curves

# -----------------------------
# Plotting functions
# -----------------------------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_reward_distributions(scenarios, cfg, outdir):
    for scenario in scenarios:
        rng = np.random.default_rng(cfg["seed"] + hash(scenario.name) % 100000)
        draws = []
        labels = []
        for k, label in enumerate(scenario.arm_labels):
            vals = [scenario.draw_fn(k, rng) for _ in range(cfg["distribution_draws"])]
            draws.append(vals)
            labels.append(label)

        plt.figure(figsize=(8, 4.8))
        plt.boxplot(draws, tick_labels=labels, showfliers=False)
        plt.yscale("log")
        plt.title(f"Reward distributions (log scale): {scenario.name}")
        plt.ylabel("Reward")
        plt.xlabel("Arm")
        savefig(os.path.join(outdir, f"{scenario.name}_reward_distributions.png"))

def plot_regret_curves(traj_df, scenarios, cfg, outdir):
    algs = cfg["algorithms"]
    for scenario in scenarios:
        sub = traj_df[traj_df["scenario"] == scenario.name]
        plt.figure(figsize=(8, 4.8))
        for alg in algs:
            s = sub[sub["algorithm"] == alg].sort_values("time")
            x = s["time"].values
            y = s["mean_regret"].values
            se = s["se_regret"].values
            plt.plot(x, y, label=alg)
            plt.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.15)
        plt.title(f"Mean cumulative regret: {scenario.name}")
        plt.xlabel("Time")
        plt.ylabel("Cumulative regret")
        plt.legend()
        savefig(os.path.join(outdir, f"{scenario.name}_mean_regret.png"))

def plot_optimal_pull_curves(traj_df, scenarios, cfg, outdir):
    algs = cfg["algorithms"]
    for scenario in scenarios:
        sub = traj_df[traj_df["scenario"] == scenario.name]
        plt.figure(figsize=(8, 4.8))
        for alg in algs:
            s = sub[sub["algorithm"] == alg].sort_values("time")
            x = s["time"].values
            y = s["mean_optimal_pull"].rolling(window=20, min_periods=1).mean() if hasattr(s["mean_optimal_pull"], "rolling") else s["mean_optimal_pull"].values
            plt.plot(x, np.asarray(y), label=alg)
        plt.title(f"Optimal-arm pull probability: {scenario.name}")
        plt.xlabel("Time")
        plt.ylabel("Probability of pulling optimal arm")
        plt.ylim(-0.02, 1.02)
        plt.legend()
        savefig(os.path.join(outdir, f"{scenario.name}_optimal_pull_probability.png"))

def plot_final_regret_boxplots(run_df, scenarios, cfg, outdir):
    algs = cfg["algorithms"]
    for scenario in scenarios:
        sub = run_df[run_df["scenario"] == scenario.name]
        data = [sub[sub["algorithm"] == alg]["final_regret"].values for alg in algs]
        plt.figure(figsize=(8, 4.8))
        plt.boxplot(data, tick_labels=algs, showfliers=False)
        plt.title(f"Final regret distribution: {scenario.name}")
        plt.ylabel("Final cumulative regret")
        savefig(os.path.join(outdir, f"{scenario.name}_final_regret_boxplot.png"))

def plot_pull_share_bars(run_df, scenarios, cfg, outdir):
    algs = cfg["algorithms"]
    for scenario in scenarios:
        sub = run_df[run_df["scenario"] == scenario.name]
        K = len(scenario.means)
        width = 0.18
        x = np.arange(K)

        plt.figure(figsize=(8, 4.8))
        for j, alg in enumerate(algs):
            vals = []
            ss = sub[sub["algorithm"] == alg]
            for k in range(K):
                vals.append(ss[f"pull_share_arm_{k+1}"].mean())
            plt.bar(x + j * width, vals, width=width, label=alg)

        plt.xticks(x + width * (len(algs)-1) / 2, scenario.arm_labels)
        plt.ylim(0, 1.0)
        plt.xlabel("Arm")
        plt.ylabel("Average pull share")
        plt.title(f"Arm allocation patterns: {scenario.name}")
        plt.legend()
        savefig(os.path.join(outdir, f"{scenario.name}_pull_share_bars.png"))

def plot_posterior_sensitivity(curves, outdir):
    grid = curves["grid"]
    plt.figure(figsize=(8, 4.8))
    plt.plot(grid, curves["gauss_base"], label="GaussianTS: base sample")
    plt.plot(grid, curves["gauss_whale"], label="GaussianTS: + one whale")
    plt.plot(grid, curves["rgts_base"], label="RGTS: base sample")
    plt.plot(grid, curves["rgts_whale"], label="RGTS: + one whale")
    plt.title("Posterior sensitivity to a single whale outlier")
    plt.xlabel("Mean parameter")
    plt.ylabel("Posterior density")
    plt.legend()
    savefig(os.path.join(outdir, "posterior_sensitivity_whale.png"))

# -----------------------------
# Summaries and rankings
# -----------------------------
def make_summary_tables(run_df, scenarios, cfg):
    summary = (
        run_df.groupby(["scenario", "algorithm"], as_index=False)
        .agg(
            mean_final_regret=("final_regret", "mean"),
            sd_final_regret=("final_regret", "std"),
            median_final_regret=("final_regret", "median"),
            mean_total_reward=("total_reward", "mean"),
            best_arm_id_rate=("best_arm_correct", "mean"),
            mean_optimal_pull_share=("optimal_pull_share", "mean"),
            mean_opt_arm_est_error=("opt_arm_est_error", "mean"),
        )
    )

    rankings = summary.sort_values(["scenario", "mean_final_regret", "mean_opt_arm_est_error", "best_arm_id_rate"],
                                   ascending=[True, True, True, False]).copy()
    rankings["scenario_rank"] = rankings.groupby("scenario")["mean_final_regret"].rank(method="dense")

    pull_cols = [c for c in run_df.columns if c.startswith("pull_share_arm_")]
    pull_share = (
        run_df.groupby(["scenario", "algorithm"], as_index=False)[pull_cols]
        .mean()
    )

    return summary, rankings, pull_share

def save_tables(run_df, traj_df, summary_df, rankings_df, pull_df, sensitivity_df, root):
    run_df.to_csv(os.path.join(root, "tables", "simulation_run_level_summary.csv"), index=False)
    traj_df.to_csv(os.path.join(root, "tables", "simulation_trajectory_summary.csv"), index=False)
    summary_df.to_csv(os.path.join(root, "tables", "simulation_summary.csv"), index=False)
    rankings_df.to_csv(os.path.join(root, "tables", "scenario_rankings.csv"), index=False)
    pull_df.to_csv(os.path.join(root, "tables", "pull_share_summary.csv"), index=False)
    sensitivity_df.to_csv(os.path.join(root, "tables", "posterior_sensitivity_summary.csv"), index=False)

# -----------------------------
# Main execution
# -----------------------------
def main(cfg=GLOBAL_CONFIG):
    ensure_dirs(SAVE_ROOT)
    with open(os.path.join(SAVE_ROOT, "metadata", "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("Starting RG-TS simulation suite")
    print(json.dumps(cfg, indent=2))

    run_df, traj_df, scenarios = run_simulation_suite(cfg)
    summary_df, rankings_df, pull_df = make_summary_tables(run_df, scenarios, cfg)
    sensitivity_df, sensitivity_curves = posterior_sensitivity_experiment(cfg)

    save_tables(run_df, traj_df, summary_df, rankings_df, pull_df, sensitivity_df, SAVE_ROOT)

    # Inline tables
    show_df(summary_df.round(4), "Simulation summary")
    show_df(rankings_df.round(4), "Scenario-wise ranking by mean final regret")
    show_df(pull_df.round(4), "Average pull-share summary")
    show_df(sensitivity_df.round(4), "Posterior sensitivity summary")

    # Figures
    fig_dir = os.path.join(SAVE_ROOT, "figures")
    plot_reward_distributions(scenarios, cfg, fig_dir)
    plot_regret_curves(traj_df, scenarios, cfg, fig_dir)
    plot_optimal_pull_curves(traj_df, scenarios, cfg, fig_dir)
    plot_final_regret_boxplots(run_df, scenarios, cfg, fig_dir)
    plot_pull_share_bars(run_df, scenarios, cfg, fig_dir)
    plot_posterior_sensitivity(sensitivity_curves, fig_dir)

    archive_path = shutil.make_archive(SAVE_ROOT, "zip", root_dir=SAVE_ROOT)
    print(f"\nAll outputs saved under: {os.path.abspath(SAVE_ROOT)}")
    print(f"Zip archive created at: {os.path.abspath(archive_path)}")

    try:
        from google.colab import files
        files.download(archive_path)
    except Exception:
        pass

if __name__ == "__main__":
    main()
