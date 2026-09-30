#!/usr/bin/env python3
"""
Market vs History: out-of-sample predictive test.

Does a team's recent history improve prediction of its next game beyond the
closing-odds implied probability? For each team-game we build features from
that team's PREVIOUS games only (no current/future information):

    prev_1_residual   last game's (result - implied probability)
    prev_3_residual   mean residual over the last 3 games (fewer if unavailable)
    prev_5_residual   mean residual over the last 5 games
    prev_win          last game won (1) or lost (0)
    streak_length     current streak, + for wins / - for losses

Model: logistic regression with logit(implied probability) as a fixed offset,
so the market forecast is the baseline and features can only adjust it:

    logit P(win) = logit(p_market) + X @ beta

Validation: expanding walk-forward. Rows are ordered by game; fold k trains on
all rows before the k-th test block and predicts that block. Metrics (log loss,
Brier) are computed on the pooled out-of-sample predictions.

The identical pipeline is run on every matched Bernoulli history, giving a null
distribution for "improvement over market alone" that arises purely by chance.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

N_FOLDS = 5
N_SIMS = 10000
FEATURE_SETS = {
    'plus_last_1': ['prev_1_residual'],
    'plus_last_3': ['prev_3_residual'],
    'plus_last_5': ['prev_5_residual'],
    'plus_all': ['prev_1_residual', 'prev_3_residual', 'prev_5_residual',
                 'prev_win', 'streak_length'],
}
EPS = 1e-10


def load_data():
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)
    mc = np.load('data/simulated/monte_carlo_outcomes.npz')['outcomes']
    return games, mc


def build_rows(games):
    """
    One row per team-game that has at least one prior game for that team.
    Returns arrays describing each row and the indices of that team's previous
    games (padded with -1), ordered by game index.
    """
    p_home = np.array([g['p_home_vig_free'] for g in games])
    order = defaultdict(list)
    for i, g in enumerate(games):
        order[g['home_team']].append((i, True))
        order[g['away_team']].append((i, False))

    rows = []  # (game_idx, is_home, prev game idxs (up to 5), prev is_home flags, all prior count)
    for team, lst in order.items():
        idx = [a for a, _ in lst]
        ih = [b for _, b in lst]
        for k in range(1, len(lst)):
            rows.append((idx[k], ih[k], idx[:k], ih[:k]))
    rows.sort(key=lambda r: r[0])

    n = len(rows)
    game = np.array([r[0] for r in rows])
    is_home = np.array([r[1] for r in rows], dtype=bool)
    p = np.where(is_home, p_home[game], 1 - p_home[game])

    # Previous 5 games (most recent first), padded with -1
    prev_idx = np.full((n, 5), -1)
    prev_home = np.zeros((n, 5), dtype=bool)
    # Full prior history for streak computation, capped at 40 games back
    MAXS = 40
    hist_idx = np.full((n, MAXS), -1)
    hist_home = np.zeros((n, MAXS), dtype=bool)
    for j, (_, _, pidx, pih) in enumerate(rows):
        rp, rh = pidx[::-1], pih[::-1]
        m5 = min(5, len(rp))
        prev_idx[j, :m5] = rp[:m5]
        prev_home[j, :m5] = rh[:m5]
        ms = min(MAXS, len(rp))
        hist_idx[j, :ms] = rp[:ms]
        hist_home[j, :ms] = rh[:ms]

    return {
        'game': game, 'is_home': is_home, 'p': p, 'p_home': p_home,
        'prev_idx': prev_idx, 'prev_home': prev_home,
        'hist_idx': hist_idx, 'hist_home': hist_home,
    }


def team_view(R, home_win, idx, home):
    """Team result (1=win) and team implied prob for game indices idx (-1 -> nan)."""
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    hw = home_win[safe].astype(float)
    res = np.where(home, hw, 1 - hw)
    ph = R['p_home'][safe]
    pt = np.where(home, ph, 1 - ph)
    res = np.where(valid, res, np.nan)
    pt = np.where(valid, pt, np.nan)
    return res, pt


def features(R, home_win):
    """Feature matrix dict and target y for one outcome history."""
    res5, p5 = team_view(R, home_win, R['prev_idx'], R['prev_home'])
    resid5 = res5 - p5
    f = {
        'prev_1_residual': resid5[:, 0],
        'prev_3_residual': np.nanmean(resid5[:, :3], axis=1),
        'prev_5_residual': np.nanmean(resid5, axis=1),
        'prev_win': res5[:, 0],
    }
    resh, _ = team_view(R, home_win, R['hist_idx'], R['hist_home'])
    last = resh[:, :1]
    same = (resh == last)
    # streak = count of leading equal results
    run = np.cumprod(np.where(np.isnan(resh), False, same), axis=1).sum(axis=1)
    f['streak_length'] = np.where(last[:, 0] == 1, run, -run)

    hw = home_win[R['game']].astype(float)
    y = np.where(R['is_home'], hw, 1 - hw)
    return f, y


def fit_offset_logit(X, y, offset, iters=25):
    """Newton-Raphson for logistic regression with fixed offset (no intercept)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = offset + X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        g = X.T @ (y - mu)
        H = (X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def log_loss(y, q):
    q = np.clip(q, EPS, 1 - EPS)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))


def brier(y, q):
    return np.mean((y - q) ** 2)


def evaluate(R, home_win):
    """Out-of-sample log loss / Brier for market-only and each feature set."""
    f, y = features(R, home_win)
    p = R['p']
    offset = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1))
    n = len(y)
    block = n // (N_FOLDS + 1)
    test_mask = np.zeros(n, dtype=bool)
    preds = {name: np.full(n, np.nan) for name in FEATURE_SETS}

    for k in range(N_FOLDS):
        tr_end = block * (k + 1)
        te_end = n if k == N_FOLDS - 1 else tr_end + block
        tr, te = slice(0, tr_end), slice(tr_end, te_end)
        test_mask[te] = True
        for name, cols in FEATURE_SETS.items():
            X = np.column_stack([f[c] for c in cols])
            beta = fit_offset_logit(X[tr], y[tr], offset[tr])
            eta = offset[te] + X[te] @ beta
            preds[name][te] = 1 / (1 + np.exp(-eta))

    yt, pt = y[test_mask], p[test_mask]
    out = {'market_only': {'log_loss': log_loss(yt, pt), 'brier': brier(yt, pt)}}
    for name in FEATURE_SETS:
        q = preds[name][test_mask]
        out[name] = {'log_loss': log_loss(yt, q), 'brier': brier(yt, q)}
    return out


def power_check(R, home_win, betas=(0.0, 0.25, 0.5, 1.0), seed=0):
    """
    Sanity check that the pipeline can detect a real effect. Plant an effect of
    size beta on the last-game residual into synthetic outcomes, then measure
    the out-of-sample improvement the '+ last game' model finds.
    """
    f, _ = features(R, home_win)
    p = R['p']
    offset = np.log(p / (1 - p))
    X = f['prev_1_residual'][:, None]
    rng = np.random.default_rng(seed)
    n = len(p)
    block = n // (N_FOLDS + 1)
    out = {}
    for b_true in betas:
        y = (rng.random(n) < 1 / (1 + np.exp(-(offset + b_true * X[:, 0])))).astype(float)
        mask = np.zeros(n, dtype=bool)
        q = np.full(n, np.nan)
        for k in range(N_FOLDS):
            tr_end = block * (k + 1)
            te_end = n if k == N_FOLDS - 1 else tr_end + block
            tr, te = slice(0, tr_end), slice(tr_end, te_end)
            mask[te] = True
            beta = fit_offset_logit(X[tr], y[tr], offset[tr])
            q[te] = 1 / (1 + np.exp(-(offset[te] + X[te] @ beta)))
        out[str(b_true)] = {
            'fitted_beta': float(beta[0]),
            'log_loss_delta': float(log_loss(y[mask], q[mask]) - log_loss(y[mask], p[mask])),
        }
    return out


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


def main():
    out_dir = Path('output/analysis')
    out_dir.mkdir(parents=True, exist_ok=True)

    games, mc = load_data()
    R = build_rows(games)
    y_real = np.array([int(g['home_win']) for g in games])
    print(f"Team-game rows with history: {len(R['game'])}")

    real = evaluate(R, y_real)
    base = real['market_only']
    print("\nReal NFL, out of sample:")
    for name, m in real.items():
        print(f"  {name:12s} log_loss={m['log_loss']:.5f}  brier={m['brier']:.5f}  "
              f"Δll={m['log_loss'] - base['log_loss']:+.5f}")

    n_sims = min(N_SIMS, mc.shape[0])
    d_ll = {name: np.zeros(n_sims) for name in FEATURE_SETS}
    d_br = {name: np.zeros(n_sims) for name in FEATURE_SETS}
    for m in range(n_sims):
        if (m + 1) % 1000 == 0:
            print(f"  simulation {m + 1}/{n_sims}")
        ev = evaluate(R, mc[m])
        for name in FEATURE_SETS:
            d_ll[name][m] = ev[name]['log_loss'] - ev['market_only']['log_loss']
            d_br[name][m] = ev[name]['brier'] - ev['market_only']['brier']

    results = {'n_sims': n_sims, 'n_rows': int(len(R['game'])),
               'real_metrics': real, 'log_loss_delta': {}, 'brier_delta': {}}
    print("\nΔ log loss vs market alone (negative = history helps):")
    for name in FEATURE_SETS:
        rl = real[name]['log_loss'] - base['log_loss']
        rb = real[name]['brier'] - base['brier']
        results['log_loss_delta'][name] = summarize(rl, d_ll[name])
        results['brier_delta'][name] = summarize(rb, d_br[name])
        s = results['log_loss_delta'][name]
        print(f"  {name:12s} real={rl:+.5f}  chance 95%=[{s['ci_lower']:+.5f}, {s['ci_upper']:+.5f}]  "
              f"pct={s['percentile']:.1f}")

    results['power_check'] = power_check(R, y_real)
    print("\nPower check (planted effect on last-game residual):")
    for b, v in results['power_check'].items():
        print(f"  beta={b}: fitted={v['fitted_beta']:+.2f}  Δ log loss={v['log_loss_delta']:+.5f}")

    with open(out_dir / 'market_vs_history_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'market_vs_history_results.json'}")


if __name__ == '__main__':
    main()
