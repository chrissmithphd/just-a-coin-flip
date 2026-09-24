#!/usr/bin/env python3
"""
Visualize Market vs History results.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def plot_market_vs_history(results_path, output_path):
    """Create visualization of market vs history results."""

    with open(results_path) as f:
        results = json.load(f)

    # Extract data
    model_names = ['plus_last_1', 'plus_last_3', 'plus_last_5', 'plus_all']
    labels = ['+Last 1', '+Last 3', '+Last 5', '+All History']

    real_deltas = [results['real']['deltas'][name] for name in model_names]

    mc_means = [results['mc']['percentiles'][name]['mc_mean'] for name in model_names]
    ci_lower = [results['mc']['percentiles'][name]['ci_lower'] for name in model_names]
    ci_upper = [results['mc']['percentiles'][name]['ci_upper'] for name in model_names]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(labels))
    width = 0.5

    # Plot MC confidence band as shaded region
    for i in range(len(labels)):
        ax.fill_between([i - width/2, i + width/2],
                        [ci_lower[i], ci_lower[i]],
                        [ci_upper[i], ci_upper[i]],
                        alpha=0.25, color='gray', zorder=1)

    # Plot real NFL bars
    colors = ['#2E5C8A' if d < 0 else '#8B4513' for d in real_deltas]
    bars = ax.bar(x, real_deltas, width, label='Real NFL',
                  color=colors, edgecolor='black', linewidth=1.5,
                  alpha=0.8, zorder=3)

    # Add MC mean markers
    ax.scatter(x, mc_means, s=100, color='red', marker='_', linewidths=3,
              label='MC Mean', zorder=4)

    # Zero line (market baseline)
    ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7,
              label='Market-Only Baseline')

    # Styling
    ax.set_ylabel('Δ Out-of-Sample Log Loss\n(relative to market alone)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Features Added to Market Probability', fontsize=13, fontweight='bold')
    ax.set_title('Does Recent History Beat the Market?\n(lower is better prediction)',
                fontsize=15, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation
    text_y = ax.get_ylim()[1] * 0.85
    ax.text(0.98, 0.05,
           'Gray bands: 95% range from pure chance\n(Monte Carlo simulations)',
           transform=ax.transAxes,
           fontsize=10, style='italic', color='#333',
           ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Add interpretation aid
    if all(d > -0.001 for d in real_deltas):  # All near zero or positive
        ax.text(0.02, 0.95,
               '← Improvement would be negative',
               transform=ax.transAxes,
               fontsize=10, style='italic', color='green',
               ha='left', va='top')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    """Generate visualization."""
    results_path = 'output/analysis/market_vs_history_results.json'
    output_path = 'output/analysis/market_vs_history.png'

    print("Creating visualization...")
    plot_market_vs_history(results_path, output_path)
    print("✓ Visualization complete")


if __name__ == '__main__':
    main()
