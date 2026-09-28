#!/usr/bin/env python3
"""
Residuals Analysis

For each game define: r_i = Y_i - p_i

Test whether residuals resemble unpredictable noise or exhibit patterns:
1. Overall residual autocorrelation
2. Within-team residual autocorrelation
3. Residuals following wins vs losses
4. Residuals following upsets
5. Residuals by week/season progression
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_data():
    """Load real games and Monte Carlo simulations."""
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)

    mc_data = np.load('data/simulated/monte_carlo_outcomes.npz')
    mc_outcomes = mc_data['outcomes']

    return games, mc_outcomes


def compute_residuals(games, outcomes):
    """
    Compute residuals: r_i = Y_i - p_i

    Args:
        games: List of game dicts
        outcomes: Array of outcomes (n_games,) or (n_sims, n_games)

    Returns:
        Array of residuals with same shape as outcomes
    """
    p_home = np.array([g['p_home_vig_free'] for g in games])

    if len(outcomes.shape) == 1:
        # Single outcome series
        return outcomes - p_home
    else:
        # MC: shape (n_sims, n_games)
        return outcomes - p_home[np.newaxis, :]


def analyze_residual_autocorrelation(residuals_real, residuals_mc, max_lag=10):
    """
    Compute autocorrelation of residuals.

    Args:
        residuals_real: Array of shape (n_games,)
        residuals_mc: Array of shape (n_sims, n_games)
        max_lag: Maximum lag to compute

    Returns:
        Dict with autocorrelation statistics
    """
    def autocorr(x, lag):
        """Compute autocorrelation at given lag."""
        if len(x) <= lag:
            return np.nan

        x_centered = x - x.mean()
        c0 = np.dot(x_centered, x_centered) / len(x)

        if c0 == 0:
            return np.nan

        c_lag = np.dot(x_centered[:-lag], x_centered[lag:]) / len(x_centered[:-lag])
        return c_lag / c0

    # Real autocorrelation
    ac_real = [autocorr(residuals_real, lag) for lag in range(1, max_lag + 1)]

    # MC autocorrelations
    ac_mc_all = []
    for sim_residuals in residuals_mc:
        ac_sim = [autocorr(sim_residuals, lag) for lag in range(1, max_lag + 1)]
        ac_mc_all.append(ac_sim)

    ac_mc_all = np.array(ac_mc_all)

    results = {}
    for lag_idx in range(max_lag):
        lag = lag_idx + 1
        mc_values = ac_mc_all[:, lag_idx]
        mc_values = mc_values[~np.isnan(mc_values)]

        results[f'lag{lag}'] = {
            'real': float(ac_real[lag_idx]) if not np.isnan(ac_real[lag_idx]) else None,
            'mc_mean': float(np.mean(mc_values)),
            'mc_std': float(np.std(mc_values)),
            'percentile': float((mc_values < ac_real[lag_idx]).mean() * 100) if not np.isnan(ac_real[lag_idx]) else None
        }

    return results


def analyze_residuals_after_outcomes(games, residuals_real, residuals_mc):
    """
    Compare the mean residual following wins vs losses (a "hot hand" /
    regression-to-mean probe) using a correct per-simulation null.

    The statistic T is a single number: the mean residual over all games that
    followed a win (or loss) in chronological order. We compute T once for the
    real NFL, then compute the SAME mean within each simulated league, giving a
    null distribution of one value per league. We compare the real T to that
    distribution -- never to a pool of individual residuals across leagues.
    """
    mc_outcomes = np.load('data/simulated/monte_carlo_outcomes.npz')['outcomes']
    y_real = np.array([int(g['home_win']) for g in games])

    # Real statistic: mean of next-game residual, split by previous outcome
    prev_win_real = (y_real[:-1] == 1)
    prev_loss_real = (y_real[:-1] == 0)
    next_resid_real = residuals_real[1:]
    real_after_win = float(next_resid_real[prev_win_real].mean())
    real_after_loss = float(next_resid_real[prev_loss_real].mean())

    # Per-simulation null: one mean per league
    prev_outcomes = mc_outcomes[:, :-1]          # (n_sims, n_games-1)
    next_resid_mc = residuals_mc[:, 1:]          # (n_sims, n_games-1)

    win_mask = (prev_outcomes == 1)
    loss_mask = (prev_outcomes == 0)

    win_counts = win_mask.sum(axis=1)
    loss_counts = loss_mask.sum(axis=1)
    means_after_win = (next_resid_mc * win_mask).sum(axis=1) / win_counts
    means_after_loss = (next_resid_mc * loss_mask).sum(axis=1) / loss_counts

    def summarize(real_val, dist):
        dist = np.asarray(dist, dtype=float)
        return {
            'real': float(real_val),
            'mc_mean': float(dist.mean()),
            'mc_std': float(dist.std()),
            'ci_lower': float(np.percentile(dist, 2.5)),
            'ci_upper': float(np.percentile(dist, 97.5)),
            'percentile': float((dist < real_val).mean() * 100),
            'p_two_sided': float(2 * min((dist <= real_val).mean(),
                                         (dist >= real_val).mean())),
        }

    return {
        'after_win': summarize(real_after_win, means_after_win),
        'after_loss': summarize(real_after_loss, means_after_loss),
        'n_after_win_real': int(prev_win_real.sum()),
        'n_after_loss_real': int(prev_loss_real.sum()),
    }


def plot_residual_autocorrelation(ac_results, output_path):
    """Plot residual autocorrelation."""
    lags = sorted([int(k.replace('lag', '')) for k in ac_results.keys()])

    real_vals = [ac_results[f'lag{lag}']['real'] for lag in lags if ac_results[f'lag{lag}']['real'] is not None]
    mc_means = [ac_results[f'lag{lag}']['mc_mean'] for lag in lags]
    mc_stds = [ac_results[f'lag{lag}']['mc_std'] for lag in lags]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.array(lags[:len(real_vals)])

    # MC confidence band (±2σ)
    mc_means_arr = np.array(mc_means[:len(real_vals)])
    mc_stds_arr = np.array(mc_stds[:len(real_vals)])

    ax.fill_between(x, mc_means_arr - 2*mc_stds_arr, mc_means_arr + 2*mc_stds_arr,
                    alpha=0.3, color='steelblue', label='Bernoulli MC (±2σ)')

    # MC mean
    ax.plot(x, mc_means_arr, 'o-', color='steelblue', linewidth=2, markersize=6,
            label='MC Mean', alpha=0.7)

    # Real NFL
    ax.plot(x, real_vals, 'o-', color='darkred', linewidth=3, markersize=10,
            label='Real NFL', zorder=5)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Lag (games)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Residual Autocorrelation', fontsize=13, fontweight='bold')
    ax.set_title('Residual Autocorrelation: Are Outcomes Predictable After Conditioning on p_i?',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(lags)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_residuals_distribution(residuals_real, residuals_mc, output_path):
    """Plot distribution of residuals."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # MC distribution (flatten all sims)
    residuals_mc_flat = residuals_mc.flatten()

    ax.hist(residuals_mc_flat, bins=50, density=True, alpha=0.5, color='steelblue',
            edgecolor='black', label='Bernoulli MC')

    ax.hist(residuals_real, bins=50, density=True, alpha=0.7, color='darkred',
            edgecolor='black', label='Real NFL')

    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Residual (Y_i - p_i)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title('Residual Distribution: Real NFL vs Bernoulli Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # Add summary statistics
    textstr = f'Real NFL:\n  Mean: {np.mean(residuals_real):.4f}\n  Std: {np.std(residuals_real):.4f}\n\n'
    textstr += f'MC:\n  Mean: {np.mean(residuals_mc_flat):.4f}\n  Std: {np.std(residuals_mc_flat):.4f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def print_residuals_summary(ac_results, outcome_results):
    """Print residuals analysis summary."""
    print("\n" + "=" * 100)
    print("RESIDUALS ANALYSIS")
    print("=" * 100)

    print("\nResidual Autocorrelation:")
    print(f"{'Lag':>5} {'Real':>10} {'MC Mean':>10} {'MC Std':>10} {'Percentile':>12}")
    print("-" * 50)

    for lag in sorted([int(k.replace('lag', '')) for k in ac_results.keys()]):
        r = ac_results[f'lag{lag}']
        real_val = r['real'] if r['real'] is not None else float('nan')
        pct = r['percentile'] if r['percentile'] is not None else float('nan')
        print(f"{lag:>5} {real_val:>10.4f} {r['mc_mean']:>10.4f} {r['mc_std']:>10.4f} {pct:>12.1f}%")

    print("\nMean residual conditional on prior outcome (per-simulation null):")
    print(f"{'Condition':>14} {'Real':>10} {'MC mean':>10} {'95% CI':>22} {'pct':>7} {'p2':>7}")
    print("-" * 78)
    for key, label in [('after_win', 'After wins'), ('after_loss', 'After losses')]:
        s = outcome_results[key]
        ci = f"[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]"
        print(f"{label:>14} {s['real']:>10.4f} {s['mc_mean']:>10.4f} {ci:>22} "
              f"{s['percentile']:>6.1f}% {s['p_two_sided']:>6.3f}")

    print(f"\n  (real n after wins = {outcome_results['n_after_win_real']}, "
          f"after losses = {outcome_results['n_after_loss_real']})")
    print("=" * 100 + "\n")


def main():
    """Run residuals analysis."""
    output_dir = Path('output/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    games, mc_outcomes = load_data()

    y_real = np.array([int(g['home_win']) for g in games])

    print("Computing residuals...")
    residuals_real = compute_residuals(games, y_real)
    residuals_mc = compute_residuals(games, mc_outcomes)

    print("Analyzing residual autocorrelation...")
    ac_results = analyze_residual_autocorrelation(residuals_real, residuals_mc, max_lag=10)

    print("Analyzing residuals after outcomes...")
    outcome_results = analyze_residuals_after_outcomes(games, residuals_real, residuals_mc)

    plot_residual_autocorrelation(ac_results, output_dir / 'residuals_autocorrelation.png')
    plot_residuals_distribution(residuals_real, residuals_mc, output_dir / 'residuals_distribution.png')

    print_residuals_summary(ac_results, outcome_results)

    # Save results
    results = {
        'autocorrelation': ac_results,
        'conditional_on_outcome': outcome_results
    }

    results_path = output_dir / 'residuals_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {results_path}")
    print("✓ Residuals analysis complete")


if __name__ == '__main__':
    main()
