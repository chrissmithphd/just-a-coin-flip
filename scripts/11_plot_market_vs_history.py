#!/usr/bin/env python3
"""Plot out-of-sample change in log loss: real NFL vs the range produced by chance."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['plus_last_1', 'plus_last_3', 'plus_last_5', 'plus_all']
LABELS = ['+ Last game', '+ Last 3 games', '+ Last 5 games', '+ All recent\nhistory']
SCALE = 1000  # show in thousandths of log loss


def main():
    with open('output/analysis/market_vs_history_results.json') as f:
        r = json.load(f)['log_loss_delta']

    real = np.array([r[n]['real'] for n in NAMES]) * SCALE
    lo = np.array([r[n]['ci_lower'] for n in NAMES]) * SCALE
    hi = np.array([r[n]['ci_upper'] for n in NAMES]) * SCALE
    x = np.arange(len(NAMES))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhspan(-100, 0, color='#e8f4ea', zorder=0)
    ax.text(3.45, lo.min() * 1.25, 'Better than the odds alone ↓', ha='right',
            va='top', fontsize=10, color='darkgreen', style='italic')
    ax.axhline(0, color='black', linewidth=1.5, label='Closing odds alone')
    ax.bar(x, hi - lo, bottom=lo, width=0.55, color='lightgray', edgecolor='gray',
           label='Range from pure chance (95%, 10,000 random histories)', zorder=2)
    ax.scatter(x, real, s=160, color='darkred', edgecolor='black', zorder=5,
               label='Real NFL')

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylabel('Change in prediction error vs. odds alone\n(log loss × 1,000; lower is better)',
                  fontsize=11, fontweight='bold')
    ax.set_title('Does Recent History Beat the Odds?\nAdding team form changes prediction accuracy by essentially nothing',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(lo.min() * 1.6, hi.max() * 1.15)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig('output/analysis/market_vs_history.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: output/analysis/market_vs_history.png')


if __name__ == '__main__':
    main()
