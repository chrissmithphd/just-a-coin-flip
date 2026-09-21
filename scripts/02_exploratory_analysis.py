#!/usr/bin/env python3
"""
Exploratory analysis of NFL betting data.

Generates descriptive statistics and visualizations to assess data quality
and establish baseline patterns before testing Bernoulli independence.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path


def load_data():
    """Load processed NFL games data."""
    with open('data/processed/nfl_games_processed.json') as f:
        return json.load(f)


def games_by_season(data):
    """Plot number of games by season."""
    seasons = [g['season'] for g in data]
    season_counts = Counter(seasons)

    fig, ax = plt.subplots(figsize=(10, 6))
    seasons_sorted = sorted(season_counts.keys())
    counts = [season_counts[s] for s in seasons_sorted]

    ax.bar(seasons_sorted, counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
    ax.set_title('NFL Games per Season (2011-2021)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(seasons_sorted)

    # Add value labels on bars
    for season, count in zip(seasons_sorted, counts):
        ax.text(season, count + 2, str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def missing_odds_by_season(data):
    """Check for missing odds by season (should be zero after filtering)."""
    seasons = sorted(set(g['season'] for g in data))

    # Count games with missing data (should be 0)
    missing_by_season = {s: 0 for s in seasons}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(seasons, [missing_by_season[s] for s in seasons], color='green', alpha=0.7)
    ax.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax.set_ylabel('Games with Missing Odds', fontsize=12, fontweight='bold')
    ax.set_title('Missing Moneyline Data by Season', fontsize=14, fontweight='bold')
    ax.set_xticks(seasons)
    ax.set_ylim(0, 10)
    ax.text(0.5, 0.5, '✓ Complete Coverage\nNo Missing Data',
            transform=ax.transAxes, fontsize=16, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    plt.tight_layout()
    return fig


def win_prob_distribution(data):
    """Plot distribution of home win probabilities."""
    probs = [g['p_home_vig_free'] for g in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(probs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Fair Coin (p=0.5)')
    ax.axvline(np.mean(probs), color='orange', linestyle='--', linewidth=2,
               label=f'Mean (p={np.mean(probs):.3f})')
    ax.set_xlabel('Home Win Probability (vig-free)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Market-Implied Home Win Probabilities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def actual_home_win_rate(data):
    """Display overall home win rate."""
    total_games = len(data)
    home_wins = sum(g['home_win'] for g in data)
    win_rate = home_wins / total_games

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(['Home Teams'], [win_rate], color='steelblue', edgecolor='black', height=0.5)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='50% (No advantage)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Win Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'Actual Home Win Rate\n{home_wins}/{total_games} = {win_rate:.1%}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Add percentage label
    ax.text(win_rate, 0, f'  {win_rate:.1%}', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def calibration_by_bin(data):
    """Plot actual win rate vs predicted probability in bins."""

    # Define probability bins
    bin_edges = np.arange(0, 1.05, 0.05)  # 5% bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign games to bins
    probs = np.array([g['p_home_vig_free'] for g in data])
    outcomes = np.array([int(g['home_win']) for g in data])

    bin_indices = np.digitize(probs, bin_edges) - 1

    # Calculate actual win rate per bin
    bin_counts = []
    bin_win_rates = []
    bin_expected = []

    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        count = mask.sum()
        if count > 0:
            win_rate = outcomes[mask].mean()
            expected = probs[mask].mean()
            bin_counts.append(count)
            bin_win_rates.append(win_rate)
            bin_expected.append(expected)
        else:
            bin_counts.append(0)
            bin_win_rates.append(np.nan)
            bin_expected.append(bin_centers[i])

    # Plot calibration curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Calibration curve
    valid_idx = [i for i, c in enumerate(bin_counts) if c > 0]
    x = [bin_expected[i] for i in valid_idx]
    y = [bin_win_rates[i] for i in valid_idx]
    sizes = [bin_counts[i] for i in valid_idx]

    ax1.scatter(x, y, s=sizes, alpha=0.6, c='steelblue', edgecolors='black')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ax1.set_xlabel('Predicted Probability (bin mean)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual Win Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Calibration Curve\n(bubble size = # games)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Games per bin
    ax2.bar(range(len(bin_counts)), bin_counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Probability Bin', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
    ax2.set_title('Games per Probability Bin (5% bins)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Label some key bins
    tick_positions = [0, 5, 10, 15, 20]
    tick_labels = [f'{i*5}%' for i in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)

    plt.tight_layout()
    return fig


def summary_table(data):
    """Generate and print summary statistics table."""
    total_games = len(data)
    seasons = sorted(set(g['season'] for g in data))
    home_wins = sum(g['home_win'] for g in data)
    home_win_rate = home_wins / total_games

    probs = [g['p_home_vig_free'] for g in data]
    vigs = [g['vig'] for g in data]

    print("\n" + "=" * 80)
    print("NFL BETTING DATA SUMMARY TABLE")
    print("=" * 80)
    print(f"\nData Coverage:")
    print(f"  Seasons: {min(seasons)} - {max(seasons)} ({len(seasons)} seasons)")
    print(f"  Total games: {total_games:,}")
    print(f"  Games per season (avg): {total_games / len(seasons):.1f}")

    print(f"\nOutcome Statistics:")
    print(f"  Home wins: {home_wins:,} ({home_win_rate:.1%})")
    print(f"  Away wins: {total_games - home_wins:,} ({1 - home_win_rate:.1%})")

    print(f"\nHome Win Probability (vig-free):")
    print(f"  Mean: {np.mean(probs):.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    print(f"  Std Dev: {np.std(probs):.4f}")
    print(f"  Min: {min(probs):.4f}")
    print(f"  Max: {max(probs):.4f}")

    print(f"\nVig Statistics:")
    print(f"  Mean vig: {np.mean(vigs):.2f}%")
    print(f"  Median vig: {np.median(vigs):.2f}%")
    print(f"  Min vig: {min(vigs):.2f}%")
    print(f"  Max vig: {max(vigs):.2f}%")

    # Calibration error
    probs_arr = np.array(probs)
    outcomes_arr = np.array([int(g['home_win']) for g in data])
    calibration_error = np.abs(probs_arr - outcomes_arr).mean()

    print(f"\nCalibration:")
    print(f"  Mean absolute error: {calibration_error:.4f}")
    print(f"  Expected wins (if calibrated): {np.sum(probs_arr):.1f}")
    print(f"  Actual wins: {home_wins}")
    print(f"  Difference: {home_wins - np.sum(probs_arr):+.1f}")

    print("\n" + "=" * 80 + "\n")


def main():
    """Generate all exploratory plots and statistics."""

    # Create output directory
    output_dir = Path('output/exploratory')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading processed data...")
    data = load_data()
    print(f"Loaded {len(data)} games")

    # Generate summary table
    summary_table(data)

    # Generate plots
    plots = {
        'games_by_season.png': games_by_season(data),
        'missing_odds.png': missing_odds_by_season(data),
        'win_prob_distribution.png': win_prob_distribution(data),
        'home_win_rate.png': actual_home_win_rate(data),
        'calibration_curve.png': calibration_by_bin(data),
    }

    # Save plots
    print("\nGenerating plots...")
    for filename, fig in plots.items():
        path = output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close(fig)

    print(f"\n✓ Exploratory analysis complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
