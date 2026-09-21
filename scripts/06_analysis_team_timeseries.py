#!/usr/bin/env python3
"""
Team Time Series Analysis

For each team, construct chronological sequences and test for:
1. Winning/losing streak distributions
2. Runs tests (too many or too few streaks)
3. Autocorrelation (do recent results predict future results?)
4. Season win-total variance
5. Conditional win probability after N consecutive wins/losses

Compare real team histories with their matched Bernoulli counterparts.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats


def load_data():
    """Load real games and Monte Carlo simulations."""
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)

    mc_data = np.load('data/simulated/monte_carlo_outcomes.npz')
    mc_outcomes = mc_data['outcomes']

    return games, mc_outcomes


def build_team_sequences(games, outcomes):
    """
    Build chronological win/loss sequences for each team.

    Args:
        games: List of game dicts
        outcomes: Array of home win indicators (n_games,) or (n_sims, n_games)

    Returns:
        Dict mapping team name to list of (game_idx, won, p_win) tuples
    """
    is_mc = len(outcomes.shape) == 2

    if is_mc:
        # For MC, we'll need to build sequences for each simulation
        # Return dict of {team: list of (game_idx, sim_outcomes, p_win)}
        team_data = defaultdict(list)

        for i, game in enumerate(games):
            home_team = game['home_team']
            away_team = game['away_team']
            p_home = game['p_home_vig_free']
            p_away = 1 - p_home

            # outcomes shape: (n_sims, n_games)
            home_outcomes = outcomes[:, i]  # shape (n_sims,)
            away_outcomes = 1 - home_outcomes

            team_data[home_team].append((i, home_outcomes, p_home))
            team_data[away_team].append((i, away_outcomes, p_away))

        return team_data

    else:
        # Real data: single outcome per game
        team_data = defaultdict(list)

        for i, game in enumerate(games):
            home_team = game['home_team']
            away_team = game['away_team']
            p_home = game['p_home_vig_free']
            p_away = 1 - p_home
            home_won = outcomes[i]

            team_data[home_team].append((i, int(home_won), p_home))
            team_data[away_team].append((i, int(1 - home_won), p_away))

        return team_data


def compute_streaks(sequence):
    """
    Compute all winning and losing streaks in a sequence.

    Args:
        sequence: Array of 0/1 outcomes

    Returns:
        Tuple of (win_streaks, loss_streaks) as lists of streak lengths
    """
    if len(sequence) == 0:
        return [], []

    win_streaks = []
    loss_streaks = []
    current_streak = 1
    current_outcome = sequence[0]

    for i in range(1, len(sequence)):
        if sequence[i] == current_outcome:
            current_streak += 1
        else:
            if current_outcome == 1:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
            current_streak = 1
            current_outcome = sequence[i]

    # Add final streak
    if current_outcome == 1:
        win_streaks.append(current_streak)
    else:
        loss_streaks.append(current_streak)

    return win_streaks, loss_streaks


def analyze_team_streaks(games, y_real, mc_outcomes):
    """
    Compare real vs MC streak distributions aggregated across all teams.
    """
    # Build team sequences
    team_data_real = build_team_sequences(games, y_real)

    # Real NFL streaks
    all_win_streaks_real = []
    all_loss_streaks_real = []

    for team, game_list in team_data_real.items():
        # Sort by game index (chronological)
        game_list = sorted(game_list, key=lambda x: x[0])
        outcomes = np.array([g[1] for g in game_list])

        win_streaks, loss_streaks = compute_streaks(outcomes)
        all_win_streaks_real.extend(win_streaks)
        all_loss_streaks_real.extend(loss_streaks)

    # MC streaks (average across simulations)
    team_data_mc = build_team_sequences(games, mc_outcomes)

    all_win_streaks_mc = []
    all_loss_streaks_mc = []

    for team, game_list in team_data_mc.items():
        game_list = sorted(game_list, key=lambda x: x[0])
        n_sims = mc_outcomes.shape[0]

        # For each simulation
        for sim_idx in range(n_sims):
            outcomes_sim = np.array([g[1][sim_idx] for g in game_list])
            win_streaks, loss_streaks = compute_streaks(outcomes_sim)
            all_win_streaks_mc.extend(win_streaks)
            all_loss_streaks_mc.extend(loss_streaks)

    return {
        'real': {
            'win_streaks': all_win_streaks_real,
            'loss_streaks': all_loss_streaks_real
        },
        'mc': {
            'win_streaks': all_win_streaks_mc,
            'loss_streaks': all_loss_streaks_mc
        }
    }


def compute_autocorrelation(sequence, max_lag=5):
    """
    Compute autocorrelation for lags 1 through max_lag.

    Args:
        sequence: Array of outcomes
        max_lag: Maximum lag to compute

    Returns:
        Array of autocorrelations [lag1, lag2, ..., lag_max]
    """
    if len(sequence) < max_lag + 1:
        return np.full(max_lag, np.nan)

    # Center the sequence
    mean = sequence.mean()
    centered = sequence - mean

    autocorrs = []
    for lag in range(1, max_lag + 1):
        if len(sequence) <= lag:
            autocorrs.append(np.nan)
            continue

        cov = np.mean(centered[:-lag] * centered[lag:])
        var = np.var(sequence)

        if var == 0:
            autocorrs.append(np.nan)
        else:
            autocorrs.append(cov / var)

    return np.array(autocorrs)


def analyze_team_autocorrelation(games, y_real, mc_outcomes, max_lag=5):
    """
    Compare autocorrelation in real vs MC team sequences.
    """
    team_data_real = build_team_sequences(games, y_real)

    # Real NFL autocorrelations
    autocorrs_real = []

    for team, game_list in team_data_real.items():
        game_list = sorted(game_list, key=lambda x: x[0])
        outcomes = np.array([g[1] for g in game_list])

        if len(outcomes) >= max_lag + 10:  # Need reasonable sample size
            ac = compute_autocorrelation(outcomes, max_lag)
            autocorrs_real.append(ac)

    autocorrs_real = np.array(autocorrs_real)
    mean_autocorr_real = np.nanmean(autocorrs_real, axis=0)

    # MC autocorrelations
    team_data_mc = build_team_sequences(games, mc_outcomes)
    n_sims = mc_outcomes.shape[0]

    autocorrs_mc_all = []

    for team, game_list in team_data_mc.items():
        game_list = sorted(game_list, key=lambda x: x[0])

        if len(game_list) < max_lag + 10:
            continue

        for sim_idx in range(n_sims):
            outcomes_sim = np.array([g[1][sim_idx] for g in game_list])
            ac = compute_autocorrelation(outcomes_sim, max_lag)
            autocorrs_mc_all.append(ac)

    autocorrs_mc_all = np.array(autocorrs_mc_all)

    # For each lag, get distribution from MC
    autocorr_distributions = {}
    for lag_idx in range(max_lag):
        lag = lag_idx + 1
        mc_values = autocorrs_mc_all[:, lag_idx]
        mc_values = mc_values[~np.isnan(mc_values)]

        autocorr_distributions[f'lag{lag}'] = {
            'real': float(mean_autocorr_real[lag_idx]),
            'mc_mean': float(np.mean(mc_values)),
            'mc_std': float(np.std(mc_values)),
            'mc_values': mc_values,
            'percentile': float((mc_values < mean_autocorr_real[lag_idx]).mean() * 100)
        }

    return autocorr_distributions


def plot_streak_distributions(streak_results, output_path):
    """Plot win streak and loss streak distributions."""
    win_real = Counter(streak_results['real']['win_streaks'])
    loss_real = Counter(streak_results['real']['loss_streaks'])

    # Normalize MC to match real scale
    n_sims = 10000
    win_mc_raw = Counter(streak_results['mc']['win_streaks'])
    loss_mc_raw = Counter(streak_results['mc']['loss_streaks'])

    # Average counts per simulation
    win_mc = {k: v / n_sims for k, v in win_mc_raw.items()}
    loss_mc = {k: v / n_sims for k, v in loss_mc_raw.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Win streaks
    max_streak = max(max(win_real.keys(), default=0), max(win_mc.keys(), default=0))
    x = range(1, min(max_streak + 1, 16))  # Cap at 15 for visibility

    real_counts = [win_real.get(i, 0) for i in x]
    mc_counts = [win_mc.get(i, 0) for i in x]

    width = 0.35
    x_pos = np.arange(len(x))

    ax1.bar(x_pos - width/2, real_counts, width, label='Real NFL',
            color='darkgreen', edgecolor='black', alpha=0.7)
    ax1.bar(x_pos + width/2, mc_counts, width, label='Bernoulli MC (avg)',
            color='steelblue', edgecolor='black', alpha=0.7)

    ax1.set_xlabel('Winning Streak Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Winning Streak Distribution\nReal NFL vs Bernoulli Model', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Loss streaks
    max_streak = max(max(loss_real.keys(), default=0), max(loss_mc.keys(), default=0))
    x = range(1, min(max_streak + 1, 16))

    real_counts = [loss_real.get(i, 0) for i in x]
    mc_counts = [loss_mc.get(i, 0) for i in x]

    x_pos = np.arange(len(x))

    ax2.bar(x_pos - width/2, real_counts, width, label='Real NFL',
            color='darkred', edgecolor='black', alpha=0.7)
    ax2.bar(x_pos + width/2, mc_counts, width, label='Bernoulli MC (avg)',
            color='steelblue', edgecolor='black', alpha=0.7)

    ax2.set_xlabel('Losing Streak Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Losing Streak Distribution\nReal NFL vs Bernoulli Model', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_autocorrelation(autocorr_results, output_path):
    """Plot autocorrelation by lag."""
    lags = sorted([int(k.replace('lag', '')) for k in autocorr_results.keys()])

    real_vals = [autocorr_results[f'lag{lag}']['real'] for lag in lags]
    mc_means = [autocorr_results[f'lag{lag}']['mc_mean'] for lag in lags]
    mc_stds = [autocorr_results[f'lag{lag}']['mc_std'] for lag in lags]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.array(lags)

    # MC mean with error bars (2 std)
    ax.errorbar(x, mc_means, yerr=[2*s for s in mc_stds], fmt='o-', color='steelblue',
                linewidth=2, markersize=8, capsize=5, label='Bernoulli MC (mean ± 2σ)', alpha=0.7)

    # Real NFL
    ax.plot(x, real_vals, 'o-', color='darkred', linewidth=3, markersize=10,
            label='Real NFL', zorder=5)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Lag (games)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Autocorrelation', fontsize=13, fontweight='bold')
    ax.set_title('Team-Level Autocorrelation: Real NFL vs Bernoulli Model', fontsize=14, fontweight='bold')
    ax.set_xticks(lags)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def print_timeseries_summary(streak_results, autocorr_results):
    """Print time series analysis summary."""
    print("\n" + "=" * 100)
    print("TEAM TIME SERIES ANALYSIS")
    print("=" * 100)

    # Streaks
    print("\nWinning Streaks:")
    win_real = streak_results['real']['win_streaks']
    win_mc = streak_results['mc']['win_streaks']

    print(f"  Real NFL: {len(win_real)} total streaks, longest = {max(win_real) if win_real else 0}")
    print(f"  MC (avg): {len(win_mc)/10000:.1f} streaks/sim, longest = {max(win_mc) if win_mc else 0}")

    print("\nLosing Streaks:")
    loss_real = streak_results['real']['loss_streaks']
    loss_mc = streak_results['mc']['loss_streaks']

    print(f"  Real NFL: {len(loss_real)} total streaks, longest = {max(loss_real) if loss_real else 0}")
    print(f"  MC (avg): {len(loss_mc)/10000:.1f} streaks/sim, longest = {max(loss_mc) if loss_mc else 0}")

    # Autocorrelation
    print(f"\nAutocorrelation (team-averaged):")
    print(f"{'Lag':>5} {'Real':>10} {'MC Mean':>10} {'MC Std':>10} {'Percentile':>12}")
    print("-" * 50)

    for lag in sorted([int(k.replace('lag', '')) for k in autocorr_results.keys()]):
        r = autocorr_results[f'lag{lag}']
        print(f"{lag:>5} {r['real']:>10.4f} {r['mc_mean']:>10.4f} {r['mc_std']:>10.4f} {r['percentile']:>12.1f}%")

    print("=" * 100 + "\n")


def main():
    """Run team time series analysis."""
    output_dir = Path('output/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    games, mc_outcomes = load_data()

    y_real = np.array([int(g['home_win']) for g in games])

    print("Analyzing streaks...")
    streak_results = analyze_team_streaks(games, y_real, mc_outcomes)

    print("Analyzing autocorrelation...")
    autocorr_results = analyze_team_autocorrelation(games, y_real, mc_outcomes, max_lag=5)

    plot_streak_distributions(streak_results, output_dir / 'team_streaks.png')
    plot_autocorrelation(autocorr_results, output_dir / 'team_autocorrelation.png')

    print_timeseries_summary(streak_results, autocorr_results)

    # Save results
    results = {
        'autocorrelation': autocorr_results,
        'streak_summary': {
            'real': {
                'win_streaks_count': len(streak_results['real']['win_streaks']),
                'loss_streaks_count': len(streak_results['real']['loss_streaks']),
                'max_win_streak': int(max(streak_results['real']['win_streaks'])) if streak_results['real']['win_streaks'] else 0,
                'max_loss_streak': int(max(streak_results['real']['loss_streaks'])) if streak_results['real']['loss_streaks'] else 0,
            }
        }
    }

    results_path = output_dir / 'team_timeseries_results.json'
    with open(results_path, 'w') as f:
        # Can't serialize arrays, so skip mc_values
        results_to_save = {
            'autocorrelation': {k: {kk: vv for kk, vv in v.items() if kk != 'mc_values'}
                              for k, v in autocorr_results.items()},
            'streak_summary': results['streak_summary']
        }
        json.dump(results_to_save, f, indent=2)

    print(f"Saved: {results_path}")
    print("✓ Team time series analysis complete")


if __name__ == '__main__':
    main()
