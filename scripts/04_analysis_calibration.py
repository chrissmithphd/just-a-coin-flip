#!/usr/bin/env python3
"""
Calibration Analysis

Compare actual win rates vs predicted probabilities across binned groups.
Test whether the market probabilities are well-calibrated using the
Monte Carlo Bernoulli distribution as the null model.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict


def load_data():
    """Load real games and Monte Carlo simulations."""
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)

    mc_data = np.load('data/simulated/monte_carlo_outcomes.npz')
    mc_outcomes = mc_data['outcomes']

    return games, mc_outcomes


def calibration_by_bins(games, mc_outcomes, bin_width=0.05):
    """
    Compute calibration statistics in probability bins.

    Args:
        games: List of real game dicts
        mc_outcomes: Array of shape (n_sims, n_games)
        bin_width: Width of probability bins (default 5%)

    Returns:
        Dict with bin statistics
    """
    # Extract data
    p_home = np.array([g['p_home_vig_free'] for g in games])
    y_real = np.array([int(g['home_win']) for g in games])

    # Define bins
    bin_edges = np.arange(0, 1 + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    # Assign games to bins
    bin_indices = np.digitize(p_home, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    results = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'bins': []
    }

    for i in range(n_bins):
        mask = bin_indices == i
        n_games_bin = mask.sum()

        if n_games_bin == 0:
            continue

        # Real NFL statistics
        p_mean = p_home[mask].mean()
        win_rate_real = y_real[mask].mean()

        # Monte Carlo statistics
        mc_outcomes_bin = mc_outcomes[:, mask]  # shape (n_sims, n_games_bin)
        win_rates_mc = mc_outcomes_bin.mean(axis=1)  # shape (n_sims,)

        mc_mean = win_rates_mc.mean()
        mc_std = win_rates_mc.std()
        mc_percentile = (win_rates_mc < win_rate_real).mean() * 100

        # 95% confidence interval from MC
        ci_lower = np.percentile(win_rates_mc, 2.5)
        ci_upper = np.percentile(win_rates_mc, 97.5)

        results['bins'].append({
            'bin_start': float(bin_edges[i]),
            'bin_end': float(bin_edges[i + 1]),
            'n_games': int(n_games_bin),
            'p_mean': float(p_mean),
            'win_rate_real': float(win_rate_real),
            'win_rate_mc_mean': float(mc_mean),
            'win_rate_mc_std': float(mc_std),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'percentile': float(mc_percentile),
            'outside_ci': bool(win_rate_real < ci_lower or win_rate_real > ci_upper)
        })

    return results


def plot_calibration(results, output_path):
    """
    Plot calibration curve with confidence intervals.

    Args:
        results: Dict from calibration_by_bins
        output_path: Where to save plot
    """
    bins = results['bins']

    if not bins:
        print("No bins to plot")
        return

    # Extract data for plotting
    p_mean = [b['p_mean'] for b in bins]
    win_rate_real = [b['win_rate_real'] for b in bins]
    win_rate_mc = [b['win_rate_mc_mean'] for b in bins]
    ci_lower = [b['ci_lower'] for b in bins]
    ci_upper = [b['ci_upper'] for b in bins]
    n_games = [b['n_games'] for b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.5, label='Perfect Calibration')

    # Monte Carlo confidence band
    ax1.fill_between(p_mean, ci_lower, ci_upper, alpha=0.3, color='steelblue',
                     label='95% CI (Bernoulli MC)')

    # Monte Carlo mean
    ax1.plot(p_mean, win_rate_mc, 'o-', color='steelblue', linewidth=2,
             markersize=8, label='Bernoulli Expected')

    # Real NFL
    sizes = np.array(n_games) * 5  # Scale for visibility
    scatter = ax1.scatter(p_mean, win_rate_real, s=sizes, alpha=0.8,
                         color='darkred', edgecolors='black', linewidth=1.5,
                         label='Real NFL', zorder=5)

    ax1.set_xlabel('Market-Implied Win Probability', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Actual Win Rate', fontsize=13, fontweight='bold')
    ax1.set_title('Calibration: Real NFL vs Bernoulli Model\n(bubble size = # games)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Residuals (real - expected)
    residuals = np.array(win_rate_real) - np.array(p_mean)

    ax2.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.bar(range(len(bins)), residuals, color=['darkred' if b['outside_ci'] else 'steelblue'
                                                 for b in bins],
            edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Probability Bin', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Residual (Actual - Expected)', fontsize=13, fontweight='bold')
    ax2.set_title('Calibration Residuals by Bin\n(red = outside 95% CI)',
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Label x-axis with probability ranges
    tick_positions = range(len(bins))
    tick_labels = [f"{b['bin_start']:.2f}-\n{b['bin_end']:.2f}" for b in bins]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def print_calibration_table(results):
    """Print calibration statistics table."""
    bins = results['bins']

    print("\n" + "=" * 100)
    print("CALIBRATION ANALYSIS")
    print("=" * 100)
    print(f"\n{'Prob Range':>15} {'N':>6} {'P(avg)':>8} {'Real':>8} {'MC Mean':>8} "
          f"{'95% CI':>18} {'%ile':>6} {'Outside?':>10}")
    print("-" * 100)

    n_outside = 0
    for b in bins:
        outside_mark = "⚠️ YES" if b['outside_ci'] else "✓ No"
        if b['outside_ci']:
            n_outside += 1

        ci_str = f"[{b['ci_lower']:.3f}, {b['ci_upper']:.3f}]"
        bin_range = f"{b['bin_start']:.2f}-{b['bin_end']:.2f}"

        print(f"{bin_range:>15} {b['n_games']:>6} {b['p_mean']:>8.3f} {b['win_rate_real']:>8.3f} "
              f"{b['win_rate_mc_mean']:>8.3f} {ci_str:>18} {b['percentile']:>6.1f} {outside_mark:>10}")

    print("-" * 100)
    print(f"\nBins outside 95% CI: {n_outside} / {len(bins)}")

    if n_outside == 0:
        print("✓ All bins consistent with Bernoulli model")
    elif n_outside <= len(bins) * 0.05:
        print("✓ Number of outlier bins consistent with expected Type I errors")
    else:
        print(f"⚠️  More outlier bins than expected by chance (p < 0.05 threshold)")

    print("=" * 100 + "\n")


def main():
    """Run calibration analysis."""
    output_dir = Path('output/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    games, mc_outcomes = load_data()

    print(f"Analyzing calibration...")
    results = calibration_by_bins(games, mc_outcomes, bin_width=0.05)

    print_calibration_table(results)

    plot_calibration(results, output_dir / 'calibration.png')

    # Save results (remove non-serializable numpy types)
    results_to_save = {
        'bin_edges': results['bin_edges'].tolist(),
        'bin_centers': results['bin_centers'].tolist(),
        'bins': results['bins']
    }
    results_path = output_dir / 'calibration_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved: {results_path}")

    print("✓ Calibration analysis complete")


if __name__ == '__main__':
    main()
