#!/usr/bin/env python3
"""
Independent verification of the team-persistence finding.

This script re-derives the lag-1 result from scratch (separate code path from
script 06) and cross-checks it with a second, more interpretable statistic:
the win-rate gap  P(win | won last) - P(win | lost last), averaged across teams.

It also (a) confirms the autocorrelation<->gap identity, and (b) reproduces the
original pooling BUG to show exactly why the effect was hidden.
"""

import json
import numpy as np
from collections import defaultdict


def load():
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)
    mc = np.load('data/simulated/monte_carlo_outcomes.npz')['outcomes']
    return games, mc


def team_sequences(games):
    """Return dict team -> ordered array of game indices, and home/away flag."""
    order = defaultdict(list)
    for i, g in enumerate(games):
        order[g['home_team']].append((i, True))
        order[g['away_team']].append((i, False))
    idx, ishome = {}, {}
    for t, lst in order.items():
        if len(lst) < 15:
            continue
        idx[t] = np.array([x[0] for x in lst])
        ishome[t] = np.array([x[1] for x in lst], dtype=bool)
    return idx, ishome


def team_real_seq(t, idx, ishome, y):
    hw = y[idx[t]]
    return np.where(ishome[t], hw, 1 - hw).astype(float)


def team_mc_mat(t, idx, ishome, mc):
    sub = mc[:, idx[t]].astype(float)
    away = ~ishome[t]
    if away.any():
        sub = sub.copy()
        sub[:, away] = 1 - sub[:, away]
    return sub  # (n_sims, L)


def lag1_autocorr_1d(x):
    x = x - x.mean()
    v = np.mean(x * x)
    return np.nan if v == 0 else np.mean(x[:-1] * x[1:]) / v


def lag1_autocorr_rows(M):
    x = M - M.mean(axis=1, keepdims=True)
    v = np.mean(x * x, axis=1)
    lc = np.mean(x[:, :-1] * x[:, 1:], axis=1)
    with np.errstate(invalid='ignore'):
        return np.where(v == 0, np.nan, lc / v)


def gap_1d(x):
    prev, nxt = x[:-1], x[1:]
    nw, nl = prev.sum(), (1 - prev).sum()
    if nw == 0 or nl == 0:
        return np.nan
    return (prev * nxt).sum() / nw - ((1 - prev) * nxt).sum() / nl


def gap_rows(M):
    prev, nxt = M[:, :-1], M[:, 1:]
    nw = prev.sum(axis=1)
    nl = (1 - prev).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        waw = np.where(nw == 0, np.nan, (prev * nxt).sum(axis=1) / nw)
        wal = np.where(nl == 0, np.nan, ((1 - prev) * nxt).sum(axis=1) / nl)
    return waw - wal


def summarize(name, real, dist):
    dist = np.asarray(dist)
    dist = dist[~np.isnan(dist)]
    pct = (dist < real).mean() * 100
    p2 = 2 * min((dist <= real).mean(), (dist >= real).mean())
    print(f"{name}")
    print(f"  real            = {real:.4f}")
    print(f"  MC mean         = {dist.mean():.4f}")
    print(f"  MC std          = {dist.std():.4f}")
    print(f"  95% CI          = [{np.percentile(dist,2.5):.4f}, {np.percentile(dist,97.5):.4f}]")
    print(f"  percentile      = {pct:.1f}%")
    print(f"  two-sided p     = {p2:.4f}")
    print()
    return pct, p2


def main():
    games, mc = load()
    y = np.array([int(g['home_win']) for g in games])
    idx, ishome = team_sequences(games)
    teams = list(idx.keys())
    n_sims = mc.shape[0]
    print(f"Teams: {len(teams)}   Simulations: {n_sims}\n")

    # ---- Real statistics (team-averaged) --------------------------------
    real_ac = np.nanmean([lag1_autocorr_1d(team_real_seq(t, idx, ishome, y)) for t in teams])
    real_gap = np.nanmean([gap_1d(team_real_seq(t, idx, ishome, y)) for t in teams])

    # ---- Identity check: autocorr vs win-rate gap, per team -------------
    print("IDENTITY CHECK  (lag-1 autocorr  vs  P(W|prevW) - P(W|prevL)), sample teams:")
    for t in teams[:5]:
        s = team_real_seq(t, idx, ishome, y)
        print(f"  {t:16s} autocorr={lag1_autocorr_1d(s):+.4f}   gap={gap_1d(s):+.4f}")
    print()

    # ---- Correct per-simulation null ------------------------------------
    ac_sum = np.zeros(n_sims); ac_cnt = np.zeros(n_sims)
    gap_sum = np.zeros(n_sims); gap_cnt = np.zeros(n_sims)
    for t in teams:
        M = team_mc_mat(t, idx, ishome, mc)
        ac = lag1_autocorr_rows(M)
        gp = gap_rows(M)
        for arr, s, c in [(ac, ac_sum, ac_cnt), (gp, gap_sum, gap_cnt)]:
            ok = ~np.isnan(arr)
            s[ok] += arr[ok]; c[ok] += 1
    ac_per_sim = ac_sum / ac_cnt
    gap_per_sim = gap_sum / gap_cnt

    print("=" * 64)
    print("CORRECT per-simulation null (one team-average per league):")
    print("=" * 64)
    summarize("Lag-1 team-averaged AUTOCORRELATION", real_ac, ac_per_sim)
    summarize("Team-averaged WIN-RATE GAP  P(W|prevW)-P(W|prevL)", real_gap, gap_per_sim)

    # ---- Reproduce the BUG: pool all team x sim values ------------------
    pooled = []
    for t in teams:
        M = team_mc_mat(t, idx, ishome, mc)
        ac = lag1_autocorr_rows(M)
        pooled.append(ac[~np.isnan(ac)])
    pooled = np.concatenate(pooled)
    print("=" * 64)
    print("BUGGED pooling (all team x simulation values in one bucket):")
    print("=" * 64)
    print(f"  pooled n         = {len(pooled):,}")
    print(f"  pooled MC mean   = {pooled.mean():.4f}")
    print(f"  pooled MC std    = {pooled.std():.4f}   <-- inflated spread")
    print(f"  real percentile in pooled dist = {(pooled < real_ac).mean()*100:.1f}%")
    print()
    print(f"  correct std      = {ac_per_sim.std():.4f}")
    print(f"  ratio (pooled/correct) = {pooled.std()/ac_per_sim.std():.1f}x")
    print()
    print("The pooled standard deviation is the wrong yardstick: it measures the")
    print("spread of INDIVIDUAL team-seasons, not the spread of the league-wide")
    print("average. Averaging ~36 teams shrinks the true null spread by ~sqrt(36)=6x,")
    print("which is why the real value looked ordinary (~68th pct) under the bug but")
    print("is extreme (~99.7th pct) under the correct per-league null.")


if __name__ == '__main__':
    main()
