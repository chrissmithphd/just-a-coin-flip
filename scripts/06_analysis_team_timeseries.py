#!/usr/bin/env python3
"""
Team Time Series Analysis

For each team, construct chronological win/loss sequences and test whether the
real NFL differs from matched Bernoulli simulations on:
1. Total winning/losing streak counts
2. Longest winning/losing streak
3. Team-averaged lag-k autocorrelation

CORRECT MONTE CARLO PROTOCOL
----------------------------
Every statistic T is defined on a complete league history. For the real NFL we
compute T_real once. For each of the M simulated leagues we compute T_m using
the identical procedure, giving a null distribution of M values. We report the
two-sided Monte Carlo p-value / percentile of T_real in that distribution.

In particular, a team-averaged statistic is averaged across teams SEPARATELY
within each simulated league (yielding one value per league), never pooled
across team x simulation. Extrema (longest streak) use the distribution of
per-league maxima, never the maximum across all leagues combined.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter


def load_data():
    """Load real games and Monte Carlo simulations."""
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)
    mc_outcomes = np.load('data/simulated/monte_carlo_outcomes.npz')['outcomes']
    return games, mc_outcomes


def build_team_index(games, min_games=15):
    """
    For each team, record the ordered game indices it played and whether it was
    home in each. Chronological order is the game order in `games` (already
    sorted by date during processing).

    Returns:
        dict team -> {'idx': np.array of game indices, 'is_home': np.array bool}
        restricted to teams with at least `min_games` games.
    """
    order = defaultdict(list)  # team -> list of (game_idx, is_home)
    for i, g in enumerate(games):
        order[g['home_team']].append((i, True))
        order[g['away_team']].append((i, False))

    team_index = {}
    for team, lst in order.items():
        if len(lst) < min_games:
            continue
        idx = np.array([t[0] for t in lst], dtype=int)
        is_home = np.array([t[1] for t in lst], dtype=bool)
        team_index[team] = {'idx': idx, 'is_home': is_home}
    return team_index


def team_outcomes_real(team, team_index, y_real):
    """Real win/loss sequence (0/1) for a team, chronological."""
    info = team_index[team]
    home_win = y_real[info['idx']]
    # team won iff (home and home_win) or (away and not home_win)
    return np.where(info['is_home'], home_win, 1 - home_win).astype(np.int8)


def team_outcomes_mc(team, team_index, mc_outcomes):
    """
    Win/loss matrix (n_sims, n_team_games) for a team across all simulations.
    """
    info = team_index[team]
    sub = mc_outcomes[:, info['idx']]           # (n_sims, n_team_games)
    # flip columns where the team was the away side
    away = ~info['is_home']
    if away.any():
        sub = sub.copy()
        sub[:, away] = 1 - sub[:, away]
    return sub


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def autocorr_1d(seq, lag):
    """Lag-k autocorrelation of a single 0/1 sequence."""
    if len(seq) <= lag:
        return np.nan
    x = seq - seq.mean()
    var = np.mean(x * x)
    if var == 0:
        return np.nan
    return np.mean(x[:-lag] * x[lag:]) / var


def autocorr_rows(mat, lag):
    """
    Lag-k autocorrelation for each row of a (n_sims, L) matrix.
    Returns array (n_sims,), NaN where variance is 0.
    """
    if mat.shape[1] <= lag:
        return np.full(mat.shape[0], np.nan)
    x = mat - mat.mean(axis=1, keepdims=True)
    var = np.mean(x * x, axis=1)
    lagcov = np.mean(x[:, :-lag] * x[:, lag:], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ac = np.where(var == 0, np.nan, lagcov / var)
    return ac


def analyze_team_autocorrelation(games, y_real, mc_outcomes, team_index, max_lag=5):
    """
    Team-averaged autocorrelation, compared against the distribution of the
    same team-average computed within each simulated league.
    """
    teams = list(team_index.keys())
    n_sims = mc_outcomes.shape[0]

    # Real: average across teams of each team's autocorr
    real_by_lag = {lag: [] for lag in range(1, max_lag + 1)}
    for team in teams:
        seq = team_outcomes_real(team, team_index, y_real)
        for lag in range(1, max_lag + 1):
            real_by_lag[lag].append(autocorr_1d(seq, lag))
    real_mean = {lag: np.nanmean(real_by_lag[lag]) for lag in real_by_lag}

    # MC: for each lag, accumulate sum and count across teams PER simulation,
    # then divide -> one team-average per simulation.
    sum_by_lag = {lag: np.zeros(n_sims) for lag in range(1, max_lag + 1)}
    cnt_by_lag = {lag: np.zeros(n_sims) for lag in range(1, max_lag + 1)}
    for team in teams:
        mat = team_outcomes_mc(team, team_index, mc_outcomes)
        for lag in range(1, max_lag + 1):
            ac = autocorr_rows(mat, lag)          # (n_sims,)
            valid = ~np.isnan(ac)
            sum_by_lag[lag][valid] += ac[valid]
            cnt_by_lag[lag][valid] += 1

    results = {}
    for lag in range(1, max_lag + 1):
        with np.errstate(invalid='ignore'):
            per_sim = np.where(cnt_by_lag[lag] > 0,
                               sum_by_lag[lag] / cnt_by_lag[lag], np.nan)
        per_sim = per_sim[~np.isnan(per_sim)]
        r = real_mean[lag]
        results[f'lag{lag}'] = {
            'real': float(r),
            'mc_mean': float(per_sim.mean()),
            'mc_std': float(per_sim.std()),
            'ci_lower': float(np.percentile(per_sim, 2.5)),
            'ci_upper': float(np.percentile(per_sim, 97.5)),
            'percentile': float((per_sim < r).mean() * 100),
            'p_two_sided': float(2 * min((per_sim <= r).mean(), (per_sim >= r).mean())),
        }
    return results


# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------

def streak_stats(seq):
    """
    Given a 0/1 sequence, return (n_win_streaks, n_loss_streaks,
    max_win_streak, max_loss_streak).
    """
    if len(seq) == 0:
        return 0, 0, 0, 0
    n_win = n_loss = 0
    max_win = max_loss = 0
    cur = seq[0]
    run = 1
    for v in seq[1:]:
        if v == cur:
            run += 1
        else:
            if cur == 1:
                n_win += 1
                max_win = max(max_win, run)
            else:
                n_loss += 1
                max_loss = max(max_loss, run)
            cur = v
            run = 1
    if cur == 1:
        n_win += 1
        max_win = max(max_win, run)
    else:
        n_loss += 1
        max_loss = max(max_loss, run)
    return n_win, n_loss, max_win, max_loss


def analyze_team_streaks(games, y_real, mc_outcomes, team_index):
    """
    League-level streak statistics with correct per-simulation null.

    Real values (summed/maxed across teams) are compared against the
    distribution of the same quantity computed within each simulated league.
    """
    teams = list(team_index.keys())
    n_sims = mc_outcomes.shape[0]

    # Real totals across teams
    r_nwin = r_nloss = r_maxwin = r_maxloss = 0
    win_len_hist = Counter()
    loss_len_hist = Counter()
    for team in teams:
        seq = team_outcomes_real(team, team_index, y_real)
        nw, nl, mw, ml = streak_stats(seq)
        r_nwin += nw
        r_nloss += nl
        r_maxwin = max(r_maxwin, mw)
        r_maxloss = max(r_maxloss, ml)

    # Per-simulation accumulators
    nwin = np.zeros(n_sims); nloss = np.zeros(n_sims)
    maxwin = np.zeros(n_sims); maxloss = np.zeros(n_sims)
    for team in teams:
        mat = team_outcomes_mc(team, team_index, mc_outcomes)
        for m in range(n_sims):
            nw, nl, mw, ml = streak_stats(mat[m])
            nwin[m] += nw
            nloss[m] += nl
            if mw > maxwin[m]:
                maxwin[m] = mw
            if ml > maxloss[m]:
                maxloss[m] = ml

    def summarize(real_val, dist, higher_is_extreme=None):
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
        'win_streak_count': summarize(r_nwin, nwin),
        'loss_streak_count': summarize(r_nloss, nloss),
        'max_win_streak': summarize(r_maxwin, maxwin),
        'max_loss_streak': summarize(r_maxloss, maxloss),
        '_mc_max_win_dist': maxwin,
        '_mc_max_loss_dist': maxloss,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_max_streak_distributions(streak_results, output_path):
    """Histogram of per-simulation longest streaks with the real value marked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, key, dist_key, color, title in [
        (ax1, 'max_win_streak', '_mc_max_win_dist', 'darkgreen', 'Longest Winning Streak'),
        (ax2, 'max_loss_streak', '_mc_max_loss_dist', 'darkred', 'Longest Losing Streak'),
    ]:
        dist = streak_results[dist_key]
        real = streak_results[key]['real']
        pct = streak_results[key]['percentile']
        bins = np.arange(dist.min() - 0.5, dist.max() + 1.5, 1)
        ax.hist(dist, bins=bins, color='steelblue', edgecolor='black', alpha=0.7,
                label='Per-league maximum (10,000 sims)')
        ax.axvline(real, color=color, linestyle='-', linewidth=3,
                   label=f'Real NFL: {real:.0f}  (pct {pct:.1f}%)')
        ax.set_xlabel('Longest streak (games)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of simulated leagues', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_streak_count_distributions(streak_results, output_path):
    """Histogram of per-simulation total streak counts with real value marked."""
    # Rebuild count distributions from summaries is not possible; recompute means
    # via stored dists is only for maxima. For counts we mark real vs mc mean/CI.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, color, title in [
        (ax1, 'win_streak_count', 'darkgreen', 'Total Winning-Streak Count'),
        (ax2, 'loss_streak_count', 'darkred', 'Total Losing-Streak Count'),
    ]:
        s = streak_results[key]
        # Approximate the null with a normal for display using mc_mean/mc_std
        lo, hi = s['ci_lower'], s['ci_upper']
        ax.axvspan(lo, hi, alpha=0.25, color='steelblue', label='95% CI (10,000 sims)')
        ax.axvline(s['mc_mean'], color='black', linestyle=':', linewidth=2,
                   label=f"MC mean: {s['mc_mean']:.0f}")
        ax.axvline(s['real'], color=color, linestyle='-', linewidth=3,
                   label=f"Real NFL: {s['real']:.0f}  (pct {s['percentile']:.1f}%)")
        ax.set_xlabel('Total streak count across all teams', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_autocorrelation(autocorr_results, output_path):
    """Plot team-averaged autocorrelation by lag with per-simulation 95% band."""
    lags = sorted(int(k.replace('lag', '')) for k in autocorr_results)
    real_vals = [autocorr_results[f'lag{l}']['real'] for l in lags]
    ci_lo = [autocorr_results[f'lag{l}']['ci_lower'] for l in lags]
    ci_hi = [autocorr_results[f'lag{l}']['ci_upper'] for l in lags]
    mc_mean = [autocorr_results[f'lag{l}']['mc_mean'] for l in lags]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.array(lags)
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.3, color='steelblue',
                    label='95% CI (per-league team average)')
    ax.plot(x, mc_mean, 'o-', color='steelblue', linewidth=2, markersize=6,
            label='MC mean', alpha=0.8)
    ax.plot(x, real_vals, 'o-', color='darkred', linewidth=3, markersize=10,
            label='Real NFL', zorder=5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Lag (games)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Team-averaged autocorrelation', fontsize=13, fontweight='bold')
    ax.set_title('Team-Level Autocorrelation: Real NFL vs Bernoulli Model',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(lags)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary(streak_results, autocorr_results):
    print("\n" + "=" * 92)
    print("TEAM TIME SERIES ANALYSIS (per-simulation null)")
    print("=" * 92)

    print("\nStreaks (real value vs distribution across 10,000 simulated leagues):")
    print(f"{'Statistic':>22} {'Real':>8} {'MC mean':>10} {'95% CI':>20} {'pct':>7} {'p2':>7}")
    print("-" * 92)
    for key, label in [
        ('win_streak_count', 'Win-streak count'),
        ('loss_streak_count', 'Loss-streak count'),
        ('max_win_streak', 'Longest win streak'),
        ('max_loss_streak', 'Longest loss streak'),
    ]:
        s = streak_results[key]
        ci = f"[{s['ci_lower']:.0f}, {s['ci_upper']:.0f}]"
        print(f"{label:>22} {s['real']:>8.0f} {s['mc_mean']:>10.1f} {ci:>20} "
              f"{s['percentile']:>6.1f}% {s['p_two_sided']:>6.3f}")

    print("\nTeam-averaged autocorrelation:")
    print(f"{'Lag':>5} {'Real':>10} {'MC mean':>10} {'95% CI':>22} {'pct':>7} {'p2':>7}")
    print("-" * 92)
    for lag in sorted(int(k.replace('lag', '')) for k in autocorr_results):
        r = autocorr_results[f'lag{lag}']
        ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        print(f"{lag:>5} {r['real']:>10.4f} {r['mc_mean']:>10.4f} {ci:>22} "
              f"{r['percentile']:>6.1f}% {r['p_two_sided']:>6.3f}")
    print("=" * 92 + "\n")


def main():
    output_dir = Path('output/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    games, mc_outcomes = load_data()
    y_real = np.array([int(g['home_win']) for g in games], dtype=np.int8)

    print("Indexing teams...")
    team_index = build_team_index(games, min_games=15)
    print(f"  {len(team_index)} teams, {mc_outcomes.shape[0]} simulations")

    print("Analyzing streaks (per-simulation)...")
    streak_results = analyze_team_streaks(games, y_real, mc_outcomes, team_index)

    print("Analyzing autocorrelation (per-simulation team average)...")
    autocorr_results = analyze_team_autocorrelation(games, y_real, mc_outcomes,
                                                    team_index, max_lag=5)

    plot_streak_count_distributions(streak_results, output_dir / 'team_streak_counts.png')
    plot_max_streak_distributions(streak_results, output_dir / 'team_max_streaks.png')
    plot_autocorrelation(autocorr_results, output_dir / 'team_autocorrelation.png')

    print_summary(streak_results, autocorr_results)

    # Save (drop raw distribution arrays)
    streak_save = {k: v for k, v in streak_results.items() if not k.startswith('_')}
    with open(output_dir / 'team_timeseries_results.json', 'w') as f:
        json.dump({'autocorrelation': autocorr_results, 'streaks': streak_save},
                  f, indent=2)
    print(f"Saved: {output_dir / 'team_timeseries_results.json'}")
    print("✓ Team time series analysis complete")


if __name__ == '__main__':
    main()
