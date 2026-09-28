#!/usr/bin/env python3
"""
Betting Implications

If (1) the market's vig-free probability is the true win probability, and
(2) nothing beyond the price predicts outcomes, then every bet is priced at
slightly worse than fair odds. Expected value is negative on every wager,
and no selection strategy escapes the vig.

This script demonstrates that using the real 2011-2021 data and the matched
Bernoulli Monte Carlo histories. We simulate simple flat-stake strategies and
show that both the real NFL and the "perfectly random" simulated NFLs lose
money at roughly the hold percentage.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

STAKE = 100.0


def load_data():
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)
    mc = np.load('data/simulated/monte_carlo_outcomes.npz')['outcomes']
    return games, mc


def profit_if_win(moneyline, stake=STAKE):
    """Profit on a winning bet of `stake` at American `moneyline`."""
    m = float(moneyline)
    if m < 0:
        return stake * 100.0 / abs(m)
    else:
        return stake * m / 100.0


def bet_on_home(games):
    """
    Per game, define the bet we place if we bet the home team.
    Returns arrays: stake_mask (1 if we bet), payout_if_home_win, is_home_bet.
    We'll define strategies by which games we choose to bet.
    """
    pass


def simulate_strategy(games, outcomes, selector, side):
    """
    Simulate flat $100 betting on a strategy over one outcome sequence.

    Args:
        games: list of game dicts
        outcomes: array (n_games,) of home-win indicators (0/1)
        selector: function(game) -> bool, whether to bet this game
        side: 'home', 'away', or 'favorite' -> which team we back

    Returns:
        cumulative profit array over the bet sequence, and total profit
    """
    profits = []
    for i, g in enumerate(games):
        if not selector(g):
            continue
        home_win = int(outcomes[i])

        if side == 'home':
            back_home = True
            ml = g['home_moneyline']
        elif side == 'away':
            back_home = False
            ml = g['away_moneyline']
        elif side == 'favorite':
            back_home = g['p_home_vig_free'] > 0.5
            ml = g['home_moneyline'] if back_home else g['away_moneyline']
        elif side == 'underdog':
            back_home = g['p_home_vig_free'] < 0.5
            ml = g['home_moneyline'] if back_home else g['away_moneyline']
        else:
            raise ValueError(side)

        won = (home_win == 1) if back_home else (home_win == 0)
        if won:
            profits.append(profit_if_win(ml))
        else:
            profits.append(-STAKE)

    profits = np.array(profits)
    return np.cumsum(profits), profits.sum(), len(profits)


def mc_totals(games, mc, selector, side, n_sims=10000):
    """Total profit for the strategy across each Monte Carlo history."""
    totals = []
    n = min(n_sims, mc.shape[0])
    for m in range(n):
        _, total, _ = simulate_strategy(games, mc[m], selector, side)
        totals.append(total)
    return np.array(totals)


# --- Strategy selectors ---------------------------------------------------

def sel_all(g):
    return True

def sel_coinflip(g):
    p = g['p_home_vig_free']
    return 0.45 <= p <= 0.55

def sel_heavy_fav(g):
    p = g['p_home_vig_free']
    return max(p, 1 - p) >= 0.75


STRATEGIES = [
    ('Bet every favorite',      sel_all,        'favorite'),
    ('Bet coin-flip games (45-55%)', sel_coinflip, 'favorite'),
    ('Bet heavy favorites (>=75%)',  sel_heavy_fav, 'favorite'),
    ('Bet every underdog',      sel_all,        'underdog'),
]


def main():
    out_dir = Path('output/analysis')
    out_dir.mkdir(parents=True, exist_ok=True)

    games, mc = load_data()
    y_real = np.array([int(g['home_win']) for g in games])

    print("=" * 74)
    print("BETTING IMPLICATIONS: flat $100 stake, 2011-2021")
    print("=" * 74)

    results = {}
    for name, selector, side in STRATEGIES:
        cum_real, total_real, n_bets = simulate_strategy(games, y_real, selector, side)
        totals_mc = mc_totals(games, mc, selector, side, n_sims=10000)

        turnover = n_bets * STAKE
        roi_real = total_real / turnover * 100
        pct = (totals_mc < total_real).mean() * 100

        results[name] = {
            'n_bets': int(n_bets),
            'turnover': turnover,
            'total_real': float(total_real),
            'roi_real_pct': float(roi_real),
            'mc_mean': float(totals_mc.mean()),
            'mc_std': float(totals_mc.std()),
            'mc_roi_pct': float(totals_mc.mean() / turnover * 100),
            'percentile_real': float(pct),
            'cum_real': cum_real.tolist(),
        }

        print(f"\n{name}")
        print(f"  Bets: {n_bets:,}   Turnover: ${turnover:,.0f}")
        print(f"  Real NFL profit:   ${total_real:>10,.0f}   (ROI {roi_real:+.2f}%)")
        print(f"  Bernoulli MC mean: ${totals_mc.mean():>10,.0f}   (ROI {totals_mc.mean()/turnover*100:+.2f}%)")
        print(f"  Real percentile in MC: {pct:.1f}%")

    # Save numeric results (without heavy trajectories in the summary file)
    summary = {k: {kk: vv for kk, vv in v.items() if kk != 'cum_real'}
               for k, v in results.items()}
    with open(out_dir / 'betting_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Figure 1: bankroll trajectories (real) --------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#8B4513', '#2E5C8A', '#4B0082', '#B8860B']
    for (name, _, _), c in zip(STRATEGIES, colors):
        cum = np.array(results[name]['cum_real'])
        ax.plot(range(1, len(cum) + 1), cum, linewidth=2, color=c, label=name)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Bet number (chronological, 2011-2021)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative profit ($100 flat stake)', fontsize=12, fontweight='bold')
    ax.set_title('Every Simple Strategy Bleeds Out\nReal NFL results, flat $100 bets',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'betting_bankroll.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'betting_bankroll.png'}")

    # ---- Figure 2: real vs MC distribution for the headline strategy -----
    name = 'Bet every favorite'
    selector, side = sel_all, 'favorite'
    totals_mc = mc_totals(games, mc, selector, side, n_sims=10000)
    total_real = results[name]['total_real']

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(totals_mc, bins=60, color='steelblue', edgecolor='black', alpha=0.7,
            label='10,000 random NFLs (Bernoulli)')
    ax.axvline(0, color='green', linestyle='--', linewidth=2,
               label='Break even ($0)')
    ax.axvline(total_real, color='darkred', linestyle='-', linewidth=3,
               label=f'Real NFL: ${total_real:,.0f}')
    ax.axvline(totals_mc.mean(), color='black', linestyle=':', linewidth=2,
               label=f'MC mean: ${totals_mc.mean():,.0f}')
    ax.set_xlabel('Total profit after betting every favorite ($100 flat)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of simulated NFLs', fontsize=12, fontweight='bold')
    ax.set_title('You Cannot Bet Your Way Above the Vig\nProfit betting every favorite, 2011-2021',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'betting_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_dir / 'betting_distribution.png'}")

    print("\n" + "=" * 74)
    print("Both real and simulated NFLs cluster below $0. The vig is the house edge;")
    print("randomness beyond the price means no selection rule climbs back above it.")
    print("=" * 74)


if __name__ == '__main__':
    main()
