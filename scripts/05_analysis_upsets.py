#!/usr/bin/env python3
"""
Upsets Analysis

Test whether actual upset frequency differs from what Bernoulli probabilities predict.

Define upset: The team with p < 0.5 wins (underdog victory).
For each game, upset_probability = 1 - max(p_i, 1-p_i)

Compare:
1. Total upset count in real NFL vs MC distribution
2. Upset rates by probability bins
3. Most surprising upsets
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def load_data():
    """Load real games and Monte Carlo simulations."""
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)

    mc_data = np.load('data/simulated/monte_carlo_outcomes.npz')
    mc_outcomes = mc_data['outcomes']

    return games, mc_outcomes


def identify_upsets(games):
    """
    Identify upsets in real NFL.

    Args:
        games: List of game dicts

    Returns:
        Tuple of (upset_mask, upset_prob, favorite_won_mask)
    """
    p_home = np.array([g['p_home_vig_free'] for g in games])
    y_real = np.array([int(g['home_win']) for g in games])

    # Favorite is team with p > 0.5
    home_favorite = p_home > 0.5
    favorite_won = (home_favorite & (y_real == 1)) | (~home_favorite & (y_real == 0))

    # Upset = favorite lost
    upset_occurred = ~favorite_won

    # Upset probability = probability that underdog wins
    p_favorite = np.where(home_favorite, p_home, 1 - p_home)
    upset_prob = 1 - p_favorite

    return upset_occurred, upset_prob, favorite_won


def upset_counts_by_bin(games, mc_outcomes):
    """
    Compare upset rates by probability bins.

    Bins based on favorite's win probability.
    """
    p_home = np.array([g['p_home_vig_free'] for g in games])
    y_real = np.array([int(g['home_win']) for g in games])

    # Compute favorite probabilities
    home_favorite = p_home > 0.5
    p_favorite = np.where(home_favorite, p_home, 1 - p_home)

    # Identify upsets in real data
    favorite_won_real = (home_favorite & (y_real == 1)) | (~home_favorite & (y_real == 0))
    upset_real = ~favorite_won_real

    # Define bins by favorite's probability
    bin_definitions = [
        (0.50, 0.55, '50-55%'),
        (0.55, 0.60, '55-60%'),
        (0.60, 0.65, '60-65%'),
        (0.65, 0.70, '65-70%'),
        (0.70, 0.75, '70-75%'),
        (0.75, 0.80, '75-80%'),
        (0.80, 0.90, '80-90%'),
        (0.90, 1.00, '90%+'),
    ]

    results = []

    for bin_min, bin_max, bin_label in bin_definitions:
        # Games in this bin
        in_bin = (p_favorite >= bin_min) & (p_favorite < bin_max)
        n_games = in_bin.sum()

        if n_games == 0:
            continue

        # Real upset rate
        upset_rate_real = upset_real[in_bin].mean()
        upset_count_real = upset_real[in_bin].sum()

        # Expected upset rate
        expected_upset_prob = (1 - p_favorite[in_bin]).mean()

        # Monte Carlo upset rates
        # Favorite won in MC if: (home_favorite & mc_won) | (~home_favorite & ~mc_won)
        mc_outcomes_bin = mc_outcomes[:, in_bin]  # shape (n_sims, n_games_bin)
        home_favorite_bin = home_favorite[in_bin]

        # For each simulation, count upsets
        upset_counts_mc = []
        for sim_outcomes in mc_outcomes_bin:
            favorite_won_sim = (home_favorite_bin & (sim_outcomes == 1)) | (~home_favorite_bin & (sim_outcomes == 0))
            upset_counts_mc.append((~favorite_won_sim).sum())

        upset_counts_mc = np.array(upset_counts_mc)
        upset_rates_mc = upset_counts_mc / n_games

        mc_mean_rate = upset_rates_mc.mean()
        mc_std_rate = upset_rates_mc.std()

        # Percentile of real upset rate
        percentile = (upset_rates_mc < upset_rate_real).mean() * 100

        # 95% CI
        ci_lower = np.percentile(upset_rates_mc, 2.5)
        ci_upper = np.percentile(upset_rates_mc, 97.5)
        outside_ci = (upset_rate_real < ci_lower) or (upset_rate_real > ci_upper)

        results.append({
            'bin_label': bin_label,
            'bin_min': bin_min,
            'bin_max': bin_max,
            'n_games': int(n_games),
            'expected_upset_prob': float(expected_upset_prob),
            'upset_count_real': int(upset_count_real),
            'upset_rate_real': float(upset_rate_real),
            'upset_rate_mc_mean': float(mc_mean_rate),
            'upset_rate_mc_std': float(mc_std_rate),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'percentile': float(percentile),
            'outside_ci': bool(outside_ci)
        })

    return results


def plot_upsets_by_bin(results, output_path):
    """Plot upset rates by probability bin."""
    if not results:
        return

    bin_labels = [r['bin_label'] for r in results]
    upset_rate_real = [r['upset_rate_real'] for r in results]
    expected = [r['expected_upset_prob'] for r in results]
    mc_mean = [r['upset_rate_mc_mean'] for r in results]
    ci_lower = [r['ci_lower'] for r in results]
    ci_upper = [r['ci_upper'] for r in results]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(bin_labels))
    width = 0.3

    # Expected (theoretical)
    ax.bar(x - width, expected, width, label='Theoretical (1 - p_favorite)',
           color='lightgray', edgecolor='black', alpha=0.7)

    # MC mean (should match theoretical closely)
    ax.bar(x, mc_mean, width, label='Bernoulli MC Mean',
           color='steelblue', edgecolor='black', alpha=0.7)

    # Real NFL
    colors = ['darkred' if r['outside_ci'] else 'darkgreen' for r in results]
    ax.bar(x + width, upset_rate_real, width, label='Real NFL',
           color=colors, edgecolor='black', alpha=0.8)

    # Error bars for 95% CI
    yerr_lower = np.array(mc_mean) - np.array(ci_lower)
    yerr_upper = np.array(ci_upper) - np.array(mc_mean)
    ax.errorbar(x, mc_mean, yerr=[yerr_lower, yerr_upper],
                fmt='none', ecolor='black', capsize=5, alpha=0.5, linewidth=2)

    ax.set_xlabel('Favorite Win Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Upset Rate (Underdog Wins)', fontsize=13, fontweight='bold')
    ax.set_title('Upset Frequency by Favorite Strength\nReal NFL vs Bernoulli Model (95% CI)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_upset_distribution(games, mc_outcomes, output_path):
    """Plot distribution of total upset counts."""
    upset_occurred, _, _ = identify_upsets(games)
    upset_count_real = upset_occurred.sum()

    # Compute upsets in each MC simulation
    p_home = np.array([g['p_home_vig_free'] for g in games])
    home_favorite = p_home > 0.5

    upset_counts_mc = []
    for sim_outcomes in mc_outcomes:
        favorite_won_sim = (home_favorite & (sim_outcomes == 1)) | (~home_favorite & (sim_outcomes == 0))
        upset_count = (~favorite_won_sim).sum()
        upset_counts_mc.append(upset_count)

    upset_counts_mc = np.array(upset_counts_mc)

    # Statistics
    mc_mean = upset_counts_mc.mean()
    mc_std = upset_counts_mc.std()
    percentile = (upset_counts_mc < upset_count_real).mean() * 100

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(upset_counts_mc, bins=50, color='steelblue', edgecolor='black', alpha=0.7,
            label=f'Bernoulli MC (n=10,000)\nMean: {mc_mean:.1f} ± {mc_std:.1f}')

    ax.axvline(upset_count_real, color='darkred', linestyle='--', linewidth=3,
               label=f'Real NFL: {upset_count_real}\n(Percentile: {percentile:.1f}%)')

    ax.axvline(mc_mean, color='black', linestyle=':', linewidth=2, alpha=0.7,
               label=f'MC Mean: {mc_mean:.1f}')

    # Shade 95% CI
    ci_lower = np.percentile(upset_counts_mc, 2.5)
    ci_upper = np.percentile(upset_counts_mc, 97.5)
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='green', label=f'95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]')

    ax.set_xlabel('Total Upset Count (Underdog Wins)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency (# Simulations)', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Total Upsets: Real NFL vs Bernoulli Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")

    return {
        'upset_count_real': int(upset_count_real),
        'mc_mean': float(mc_mean),
        'mc_std': float(mc_std),
        'percentile': float(percentile),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'outside_ci': upset_count_real < ci_lower or upset_count_real > ci_upper
    }


def print_upset_summary(total_results, bin_results):
    """Print upset analysis summary."""
    print("\n" + "=" * 100)
    print("UPSETS ANALYSIS")
    print("=" * 100)

    print(f"\nTotal Upsets (Underdog Wins):")
    print(f"  Real NFL: {total_results['upset_count_real']}")
    print(f"  Bernoulli MC: {total_results['mc_mean']:.1f} ± {total_results['mc_std']:.1f}")
    print(f"  95% CI: [{total_results['ci_lower']:.0f}, {total_results['ci_upper']:.0f}]")
    print(f"  Percentile: {total_results['percentile']:.1f}%")

    if total_results['outside_ci']:
        print(f"  ⚠️  Real NFL outside 95% CI")
    else:
        print(f"  ✓ Real NFL consistent with Bernoulli model")

    print(f"\n{'Favorite Prob':>15} {'N Games':>8} {'Expected':>10} {'Real Rate':>10} "
          f"{'MC Mean':>10} {'95% CI':>22} {'%ile':>7} {'Outside?':>10}")
    print("-" * 100)

    n_outside = 0
    for r in bin_results:
        outside_mark = "⚠️ YES" if r['outside_ci'] else "✓ No"
        if r['outside_ci']:
            n_outside += 1

        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"

        print(f"{r['bin_label']:>15} {r['n_games']:>8} {r['expected_upset_prob']:>10.3f} "
              f"{r['upset_rate_real']:>10.3f} {r['upset_rate_mc_mean']:>10.3f} "
              f"{ci_str:>22} {r['percentile']:>7.1f} {outside_mark:>10}")

    print("-" * 100)
    print(f"\nBins outside 95% CI: {n_outside} / {len(bin_results)}")

    if n_outside == 0:
        print("✓ All bins consistent with Bernoulli model")
    else:
        print(f"⚠️  Some bins outside expected range")

    print("=" * 100 + "\n")


def main():
    """Run upsets analysis."""
    output_dir = Path('output/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    games, mc_outcomes = load_data()

    print("Analyzing upsets...")

    # Total upsets
    total_results = plot_upset_distribution(games, mc_outcomes,
                                           output_dir / 'upsets_total.png')

    # Upsets by bin
    bin_results = upset_counts_by_bin(games, mc_outcomes)
    plot_upsets_by_bin(bin_results, output_dir / 'upsets_by_bin.png')

    print_upset_summary(total_results, bin_results)

    # Save results (convert numpy bools to Python bools)
    results = {
        'total': {k: bool(v) if isinstance(v, np.bool_) else v for k, v in total_results.items()},
        'by_bin': [
            {k: bool(v) if isinstance(v, np.bool_) else v for k, v in bin_result.items()}
            for bin_result in bin_results
        ]
    }

    results_path = output_dir / 'upsets_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")

    print("✓ Upsets analysis complete")


if __name__ == '__main__':
    main()
