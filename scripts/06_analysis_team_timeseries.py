#!/usr/bin/env python3
"""
Team Time Series Analysis

Do NFL teams show momentum — and if so, has the betting market already priced
it in? We measure two things on each team's chronological win/loss sequence and
compare against 10,000 matched Bernoulli leagues:

1. RAW win/loss autocorrelation and streak counts. These test whether outcomes
   are more clustered than independent coin flips weighted by the closing line.

2. RESIDUAL autocorrelation, where each outcome has its own game probability
   subtracted first:  r_i = Y_i - p_i.  This is the clean test of whether
   *over-performance* predicts *future over-performance* — i.e. whether any
   momentum survives after the market's line has already reacted to results.

MONTE CARLO PROTOCOL
--------------------
Every statistic is a league-level number (team values averaged within a league,
or summed/maxed across teams). We compute it once on the real NFL and once
within each of the 10,000 simulated leagues, giving a null distribution of one
value per league. We report the percentile and two-sided Monte Carlo p-value of
the real value in that distribution. Team-averaged quantities are averaged
across teams SEPARATELY within each league — never pooled across team x sim.

KEY RESULT
----------
Raw streaks and raw autocorrelation exceed the independent-coin-flip baseline
(momentum appears real), but the residual autocorrelation does NOT: once the
market's game-by-game probability is removed, the persistence disappears. The
market has already priced the streaks in. See docs/persistence_explained.md.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

MIN_GAMES = 15
MAX_LAG = 5


def load_data():
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)
    mc_outcomes = np.load('data/simulated/monte_carlo_outcomes.npz')['outcomes']
    return games, mc_outcomes


def build_team_index(games, min_games=MIN_GAMES):
    """team -> {'idx': game indices (chronological), 'is_home': bool array}."""
    order = defaultdict(list)
    for i, g in enumerate(games):
        order[g['home_team']].append((i, True))
        order[g['away_team']].append((i, False))
    out = {}
    for team, lst in order.items():
        if len(lst) < min_games:
            continue
        out[team] = {
            'idx': np.array([t[0] for t in lst]),
            'is_home': np.array([t[1] for t in lst], dtype=bool),
        }
    return out


def team_prob(team, team_index, p_home):
    """This team's market win probability for each of its games."""
    info = team_index[team]
    return np.where(info['is_home'], p_home[info['idx']], 1 - p_home[info['idx']])


def team_outcomes_real(team, team_index, y_real):
    info = team_index[team]
    hw = y_real[info['idx']]
    return np.where(info['is_home'], hw, 1 - hw).astype(float)


def team_outcomes_mc(team, team_index, mc_outcomes):
    info = team_index[team]
    sub = mc_outcomes[:, info['idx']].astype(float)
    away = ~info['is_home']
    if away.any():
        sub = sub.copy()
        sub[:, away] = 1 - sub[:, away]
    return sub  # (n_sims, L)


# --- autocorrelation --------------------------------------------------------

def autocorr_1d(seq, lag, center=True):
    x = seq - seq.mean() if center else seq
    v = np.mean(x * x)
    if v == 0 or len(x) <= lag:
        return np.nan
    return np.mean(x[:-lag] * x[lag:]) / v


def autocorr_rows(mat, lag, center=True):
    x = mat - mat.mean(axis=1, keepdims=True) if center else mat
    v = np.mean(x * x, axis=1)
    lc = np.mean(x[:, :-lag] * x[:, lag:], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(v == 0, np.nan, lc / v)


def team_averaged_null(team_index, mc_outcomes, series_fn, lag, center):
    """
    For each simulated league, average a per-team lag-k autocorrelation across
    teams -> one value per league. `series_fn(team)` returns the (n_sims, L)
    matrix of that team's per-sim series (outcomes or residuals).
    """
    n_sims = mc_outcomes.shape[0]
    s = np.zeros(n_sims)
    c = np.zeros(n_sims)
    for team in team_index:
        ac = autocorr_rows(series_fn(team), lag, center=center)
        ok = ~np.isnan(ac)
        s[ok] += ac[ok]
        c[ok] += 1
    with np.errstate(invalid='ignore'):
        per_sim = np.where(c > 0, s / c, np.nan)
    return per_sim[~np.isnan(per_sim)]


def summarize(real, dist):
    dist = np.asarray(dist)
    return {
        'real': float(real),
        'mc_mean': float(dist.mean()),
        'mc_std': float(dist.std()),
        'ci_lower': float(np.percentile(dist, 2.5)),
        'ci_upper': float(np.percentile(dist, 97.5)),
        'percentile': float((dist < real).mean() * 100),
        'p_two_sided': float(2 * min((dist <= real).mean(), (dist >= real).mean())),
    }


def analyze_autocorrelation(team_index, y_real, p_home, mc_outcomes):
    """Raw and residual team-averaged autocorrelation, lags 1..MAX_LAG."""
    teams = list(team_index)

    # series functions
    def raw_real(t):
        return team_outcomes_real(t, team_index, y_real)

    def raw_mc(t):
        return team_outcomes_mc(t, team_index, mc_outcomes)

    def res_real(t):
        return team_outcomes_real(t, team_index, y_real) - team_prob(t, team_index, p_home)

    def res_mc(t):
        return team_outcomes_mc(t, team_index, mc_outcomes) - team_prob(t, team_index, p_home)[None, :]

    results = {'raw': {}, 'residual': {}, 'lag1_null': {}}

    for lag in range(1, MAX_LAG + 1):
        # RAW: center on team mean
        real_raw = np.nanmean([autocorr_1d(raw_real(t), lag, center=True) for t in teams])
        null_raw = team_averaged_null(team_index, mc_outcomes, raw_mc, lag, center=True)
        results['raw'][f'lag{lag}'] = summarize(real_raw, null_raw)

        # RESIDUAL: already mean ~0, do not re-center
        real_res = np.nanmean([autocorr_1d(res_real(t), lag, center=False) for t in teams])
        null_res = team_averaged_null(team_index, mc_outcomes, res_mc, lag, center=False)
        results['residual'][f'lag{lag}'] = summarize(real_res, null_res)

        if lag == 1:
            results['lag1_null'] = {'raw': null_raw, 'residual': null_res,
                                    'raw_real': real_raw, 'res_real': real_res}
    return results


# --- streaks ----------------------------------------------------------------

def streak_stats(seq):
    if len(seq) == 0:
        return 0, 0, 0, 0
    n_win = n_loss = max_win = max_loss = 0
    cur, run = seq[0], 1
    for v in seq[1:]:
        if v == cur:
            run += 1
        else:
            if cur == 1:
                n_win += 1; max_win = max(max_win, run)
            else:
                n_loss += 1; max_loss = max(max_loss, run)
            cur, run = v, 1
    if cur == 1:
        n_win += 1; max_win = max(max_win, run)
    else:
        n_loss += 1; max_loss = max(max_loss, run)
    return n_win, n_loss, max_win, max_loss


def analyze_streaks(team_index, y_real, mc_outcomes):
    teams = list(team_index)
    n_sims = mc_outcomes.shape[0]

    r_nwin = r_nloss = r_maxwin = r_maxloss = 0
    for t in teams:
        nw, nl, mw, ml = streak_stats(team_outcomes_real(t, team_index, y_real))
        r_nwin += nw; r_nloss += nl
        r_maxwin = max(r_maxwin, mw); r_maxloss = max(r_maxloss, ml)

    nwin = np.zeros(n_sims); nloss = np.zeros(n_sims)
    maxwin = np.zeros(n_sims); maxloss = np.zeros(n_sims)
    for t in teams:
        mat = team_outcomes_mc(t, team_index, mc_outcomes)
        for m in range(n_sims):
            nw, nl, mw, ml = streak_stats(mat[m])
            nwin[m] += nw; nloss[m] += nl
            if mw > maxwin[m]: maxwin[m] = mw
            if ml > maxloss[m]: maxloss[m] = ml

    return {
        'win_streak_count': summarize(r_nwin, nwin),
        'loss_streak_count': summarize(r_nloss, nloss),
        'max_win_streak': summarize(r_maxwin, maxwin),
        'max_loss_streak': summarize(r_maxloss, maxloss),
        '_maxwin_dist': maxwin,
        '_maxloss_dist': maxloss,
    }


# --- after a win vs after a loss (pooled, percentage points) -----------------

def analyze_after_win_loss(games, y_real, p_home, mc_outcomes):
    """
    Pool every consecutive pair of games for the same team. For the second game
    of each pair, compare the actual win rate with the win probability implied
    by that game's closing odds, split by whether the team won or lost the
    first game. Pooling (rather than averaging per team) avoids a small-sample
    ratio bias in the per-team version.
    """
    order = defaultdict(list)
    for i, g in enumerate(games):
        order[g['home_team']].append((i, True))
        order[g['away_team']].append((i, False))
    P, N, HP, HN = [], [], [], []
    for lst in order.values():
        idx = [a for a, _ in lst]
        ih = [b for _, b in lst]
        P += idx[:-1]; N += idx[1:]; HP += ih[:-1]; HN += ih[1:]
    P, N, HP, HN = map(np.array, (P, N, HP, HN))
    p_next = np.where(HN, p_home[N], 1 - p_home[N])

    def stats(Y):
        prev = np.where(HP, Y[:, P], 1 - Y[:, P]).astype(float)
        nxt = np.where(HN, Y[:, N], 1 - Y[:, N]).astype(float)
        w, l = prev.sum(1), (1 - prev).sum(1)
        out = {
            'win_rate_after_win': (prev * nxt).sum(1) / w,
            'win_rate_after_loss': ((1 - prev) * nxt).sum(1) / l,
            'implied_after_win': (prev * p_next).sum(1) / w,
            'implied_after_loss': ((1 - prev) * p_next).sum(1) / l,
        }
        out['beat_odds_after_win'] = out['win_rate_after_win'] - out['implied_after_win']
        out['beat_odds_after_loss'] = out['win_rate_after_loss'] - out['implied_after_loss']
        out['raw_gap'] = out['win_rate_after_win'] - out['win_rate_after_loss']
        out['beat_odds_gap'] = out['beat_odds_after_win'] - out['beat_odds_after_loss']
        return out

    real = stats(y_real[None, :])
    null = stats(mc_outcomes)
    results = {k: summarize(real[k][0], null[k]) for k in real}
    results['n_pairs'] = int(len(P))
    results['_null_raw_gap'] = null['raw_gap']
    results['_null_beat_odds_gap'] = null['beat_odds_gap']
    return results


def plot_after_win_loss(awl, output_path):
    """Left: actual vs implied next-game win rate. Right: leftover gap vs chance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={'width_ratios': [1, 1.2]})

    x = np.arange(2)
    actual = [awl['win_rate_after_win']['real'] * 100, awl['win_rate_after_loss']['real'] * 100]
    implied = [awl['implied_after_win']['real'] * 100, awl['implied_after_loss']['real'] * 100]
    ax1.bar(x - 0.2, actual, 0.4, color='darkred', edgecolor='black', label='Actual win rate')
    ax1.bar(x + 0.2, implied, 0.4, color='steelblue', edgecolor='black',
            label="Win chance implied by that game's closing odds")
    for xi, a, b in zip(x, actual, implied):
        ax1.text(xi - 0.2, a + 0.6, f'{a:.1f}%', ha='center', fontsize=11, fontweight='bold')
        ax1.text(xi + 0.2, b + 0.6, f'{b:.1f}%', ha='center', fontsize=11)
    ax1.axhline(50, color='black', linestyle=':', linewidth=1, alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Game after a WIN', 'Game after a LOSS'], fontsize=12)
    ax1.set_ylim(35, 62)
    ax1.set_ylabel('Next-game win rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Teams win more after a win —\nbut the next game's odds already reflect it',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    null = awl['_null_beat_odds_gap'] * 100
    real = awl['beat_odds_gap']['real'] * 100
    ax2.hist(null, bins=50, color='steelblue', edgecolor='black', alpha=0.7,
             label='10,000 random histories')
    ax2.axvline(real, color='darkred', linewidth=3,
                label=f'Real NFL: {real:+.1f} pts')
    ax2.axvline(0, color='black', linestyle=':', linewidth=1)
    ax2.set_xlabel('Beat-the-odds after a win  minus  after a loss (percentage points)',
                   fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of random histories', fontsize=12, fontweight='bold')
    ax2.set_title('What\'s left over after the odds: an ordinary amount',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# --- plots ------------------------------------------------------------------

def plot_raw_vs_residual(ac_results, output_path):
    """
    Centerpiece: lag-1 autocorrelation distributions.
    Left  = raw win/loss (real sits far right -> streaks look real).
    Right = residual after subtracting the market line (real falls back inside).
    """
    lag1 = ac_results['lag1_null']
    raw_null, res_null = lag1['raw'], lag1['residual']
    raw_real, res_real = lag1['raw_real'], lag1['res_real']
    raw_pct = ac_results['raw']['lag1']['percentile']
    res_pct = ac_results['residual']['lag1']['percentile']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, null, real, pct, title, sub in [
        (ax1, raw_null, raw_real, raw_pct,
         'Raw win/loss streaks',
         'Real teams cluster more than coin flips'),
        (ax2, res_null, res_real, res_pct,
         'After adjusting for the market line',
         'The clustering is gone — the line already knew'),
    ]:
        ax.hist(null, bins=50, color='steelblue', edgecolor='black', alpha=0.7,
                label='Random leagues (coin flips)')
        ax.axvline(real, color='darkred', linewidth=3,
                   label=f'Real NFL: {real:.3f}\n({pct:.1f}th percentile)')
        ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_xlabel('Lag-1 autocorrelation (team-averaged)', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n{sub}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    ax1.set_ylabel('Number of simulated leagues', fontsize=12, fontweight='bold')
    fig.suptitle('Do NFL Streaks Beat the Market?', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_autocorr_by_lag(ac_results, output_path):
    """Raw vs residual autocorrelation across lags 1..MAX_LAG."""
    lags = list(range(1, MAX_LAG + 1))
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for key, color, label in [('raw', 'darkred', 'Raw win/loss'),
                              ('residual', 'seagreen', 'Adjusted for market line')]:
        real = [ac_results[key][f'lag{l}']['real'] for l in lags]
        lo = [ac_results[key][f'lag{l}']['ci_lower'] for l in lags]
        hi = [ac_results[key][f'lag{l}']['ci_upper'] for l in lags]
        ax.plot(lags, real, 'o-', color=color, linewidth=2.5, markersize=8, label=f'{label} (real)')
        ax.fill_between(lags, lo, hi, color=color, alpha=0.15,
                        label=f'{label} — random 95% range')

    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_xlabel('Lag (games apart)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Team-averaged autocorrelation', fontsize=12, fontweight='bold')
    ax.set_title('Momentum Before and After Accounting for the Market Line',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(lags)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_max_streaks(streak_results, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key, dist_key, color, title in [
        (ax1, 'max_win_streak', '_maxwin_dist', 'darkgreen', 'Longest Winning Streak'),
        (ax2, 'max_loss_streak', '_maxloss_dist', 'darkred', 'Longest Losing Streak'),
    ]:
        dist = streak_results[dist_key]
        real = streak_results[key]['real']
        pct = streak_results[key]['percentile']
        bins = np.arange(dist.min() - 0.5, dist.max() + 1.5, 1)
        ax.hist(dist, bins=bins, color='steelblue', edgecolor='black', alpha=0.7,
                label='Per-league maximum (random)')
        ax.axvline(real, color=color, linewidth=3,
                   label=f'Real NFL: {real:.0f} ({pct:.1f}th pct)')
        ax.set_xlabel('Longest streak (games)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of simulated leagues', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary(ac_results, streak_results):
    print("\n" + "=" * 88)
    print("TEAM TIME SERIES ANALYSIS (per-simulation null)")
    print("=" * 88)

    print("\nStreaks (raw win/loss):")
    print(f"{'Statistic':>22} {'Real':>8} {'MC mean':>9} {'95% CI':>18} {'pct':>7} {'p2':>7}")
    print("-" * 88)
    for key, label in [('win_streak_count', 'Win-streak count'),
                       ('loss_streak_count', 'Loss-streak count'),
                       ('max_win_streak', 'Longest win streak'),
                       ('max_loss_streak', 'Longest loss streak')]:
        s = streak_results[key]
        ci = f"[{s['ci_lower']:.0f}, {s['ci_upper']:.0f}]"
        print(f"{label:>22} {s['real']:>8.0f} {s['mc_mean']:>9.1f} {ci:>18} "
              f"{s['percentile']:>6.1f}% {s['p_two_sided']:>6.3f}")

    for kind, header in [('raw', 'RAW win/loss autocorrelation'),
                        ('residual', 'RESIDUAL autocorrelation (market line removed)')]:
        print(f"\n{header}:")
        print(f"{'Lag':>5} {'Real':>10} {'MC mean':>10} {'95% CI':>22} {'pct':>7} {'p2':>7}")
        print("-" * 88)
        for lag in range(1, MAX_LAG + 1):
            r = ac_results[kind][f'lag{lag}']
            ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            print(f"{lag:>5} {r['real']:>10.4f} {r['mc_mean']:>10.4f} {ci:>22} "
                  f"{r['percentile']:>6.1f}% {r['p_two_sided']:>6.3f}")
    print("=" * 88 + "\n")


def main():
    output_dir = Path('output/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    games, mc_outcomes = load_data()
    y_real = np.array([int(g['home_win']) for g in games], dtype=np.int8)
    p_home = np.array([g['p_home_vig_free'] for g in games])

    print("Indexing teams...")
    team_index = build_team_index(games)
    print(f"  {len(team_index)} teams, {mc_outcomes.shape[0]} simulations")

    print("Analyzing autocorrelation (raw + residual, per-league null)...")
    ac_results = analyze_autocorrelation(team_index, y_real, p_home, mc_outcomes)

    print("Analyzing streaks (per-league null)...")
    streak_results = analyze_streaks(team_index, y_real, mc_outcomes)

    print("Analyzing next game after a win vs a loss...")
    awl = analyze_after_win_loss(games, y_real, p_home, mc_outcomes)

    plot_raw_vs_residual(ac_results, output_dir / 'team_autocorrelation.png')
    plot_autocorr_by_lag(ac_results, output_dir / 'team_autocorr_by_lag.png')
    plot_max_streaks(streak_results, output_dir / 'team_max_streaks.png')
    plot_after_win_loss(awl, output_dir / 'team_after_win_loss.png')

    print_summary(ac_results, streak_results)

    # Save (drop raw distribution arrays)
    def clean(d):
        return {k: v for k, v in d.items() if not k.startswith('_') and k != 'lag1_null'}
    payload = {
        'autocorrelation_raw': clean(ac_results['raw']),
        'autocorrelation_residual': clean(ac_results['residual']),
        'streaks': {k: v for k, v in streak_results.items() if not k.startswith('_')},
        'after_win_loss': {k: v for k, v in awl.items() if not k.startswith('_')},
    }
    with open(output_dir / 'team_timeseries_results.json', 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {output_dir / 'team_timeseries_results.json'}")
    print("✓ Team time series analysis complete")


if __name__ == '__main__':
    main()
