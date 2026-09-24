#!/usr/bin/env python3
"""
Create a simple diagram illustrating the experimental design:
Real NFL vs Bernoulli synthetic histories.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path


def create_experiment_diagram():
    """Create conceptual diagram of the experiment."""

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.3, 'The Experiment', fontsize=18, fontweight='bold',
            ha='center', va='top')

    # Real NFL path (top)
    # Game
    box1 = FancyBboxPatch((0.3, 6.5), 1.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#e8f4f8', edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.05, 7.1, 'Game i', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(1.05, 6.75, 'Patriots', fontsize=9, ha='center', va='center')
    ax.text(1.05, 6.5, 'vs Dolphins', fontsize=9, ha='center', va='center')

    # Arrow
    arrow1 = FancyArrowPatch((1.9, 7.1), (3.0, 7.1),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax.add_artist(arrow1)

    # Market probability
    box2 = FancyBboxPatch((3.0, 6.5), 1.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#fff4e6', edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(3.75, 7.3, 'Market', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(3.75, 6.9, r'$p_i = 0.65$', fontsize=12, ha='center', va='center')
    ax.text(3.75, 6.55, '(vig-free)', fontsize=8, ha='center', va='center', style='italic')

    # Arrow
    arrow2 = FancyArrowPatch((4.6, 7.1), (5.7, 7.1),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax.add_artist(arrow2)

    # Real outcome
    box3 = FancyBboxPatch((5.7, 6.5), 1.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#ffe6e6', edgecolor='darkred', linewidth=3)
    ax.add_patch(box3)
    ax.text(6.45, 7.3, 'Real', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(6.45, 6.95, 'Outcome', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(6.45, 6.6, r'$Y_i$', fontsize=12, ha='center', va='center')

    # Label
    ax.text(0.2, 7.1, 'Real NFL:', fontsize=12, ha='right', va='center',
            fontweight='bold', color='darkred')

    # Synthetic path (bottom)
    # Same game
    box4 = FancyBboxPatch((0.3, 3.8), 1.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#e8f4f8', edgecolor='black', linewidth=2)
    ax.add_patch(box4)
    ax.text(1.05, 4.4, 'Same Game', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(1.05, 4.05, 'Patriots', fontsize=9, ha='center', va='center')
    ax.text(1.05, 3.8, 'vs Dolphins', fontsize=9, ha='center', va='center')

    # Arrow
    arrow3 = FancyArrowPatch((1.9, 4.4), (3.0, 4.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax.add_artist(arrow3)

    # Same probability
    box5 = FancyBboxPatch((3.0, 3.8), 1.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#fff4e6', edgecolor='black', linewidth=2)
    ax.add_patch(box5)
    ax.text(3.75, 4.6, 'Same', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(3.75, 4.2, r'$p_i = 0.65$', fontsize=12, ha='center', va='center')
    ax.text(3.75, 3.85, '(unchanged)', fontsize=8, ha='center', va='center', style='italic')

    # Arrow
    arrow4 = FancyArrowPatch((4.6, 4.4), (5.7, 4.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax.add_artist(arrow4)

    # Bernoulli draw
    box6 = FancyBboxPatch((5.7, 3.8), 1.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#e6f2ff', edgecolor='steelblue', linewidth=3)
    ax.add_patch(box6)
    ax.text(6.45, 4.6, 'Bernoulli', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(6.45, 4.25, 'Draw', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(6.45, 3.9, r'$Y_i^* \sim \mathrm{Ber}(p_i)$', fontsize=11, ha='center', va='center')

    # Label
    ax.text(0.2, 4.4, 'Synthetic:', fontsize=12, ha='right', va='center',
            fontweight='bold', color='steelblue')

    # Key insight box
    insight_box = FancyBboxPatch((8.0, 3.3), 1.8, 4.5, boxstyle="round,pad=0.15",
                                facecolor='#f0f0f0', edgecolor='black', linewidth=2,
                                linestyle='--', alpha=0.7)
    ax.add_patch(insight_box)

    ax.text(8.9, 7.5, 'What We Keep:', fontsize=10, ha='center', va='top',
            fontweight='bold')
    ax.text(8.9, 7.15, '• Teams', fontsize=9, ha='center', va='top')
    ax.text(8.9, 6.85, '• Date/season', fontsize=9, ha='center', va='top')
    ax.text(8.9, 6.55, '• Probability $p_i$', fontsize=9, ha='center', va='top')
    ax.text(8.9, 6.25, '• Game order', fontsize=9, ha='center', va='top')

    ax.text(8.9, 5.7, 'What We Replace:', fontsize=10, ha='center', va='top',
            fontweight='bold')
    ax.text(8.9, 5.35, 'Actual winner →', fontsize=9, ha='center', va='top')
    ax.text(8.9, 5.05, 'Random draw', fontsize=9, ha='center', va='top')

    ax.text(8.9, 4.4, 'Result:', fontsize=10, ha='center', va='top',
            fontweight='bold')
    ax.text(8.9, 4.05, '10,000 synthetic', fontsize=9, ha='center', va='top')
    ax.text(8.9, 3.75, 'NFL histories', fontsize=9, ha='center', va='top')
    ax.text(8.9, 3.45, 'with same $p_i$', fontsize=9, ha='center', va='top')

    # Question
    ax.text(5, 1.8, 'Can you tell which NFL is real?', fontsize=14,
            ha='center', va='center', fontweight='bold', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    # Bottom note
    ax.text(5, 0.5, 'Each game keeps its real pregame probability; only the outcome is replaced by a Bernoulli draw',
            fontsize=9, ha='center', va='center', style='italic', color='#555')

    plt.tight_layout()
    return fig


def main():
    """Generate experiment diagram."""
    output_dir = Path('output/diagrams')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating experiment diagram...")
    fig = create_experiment_diagram()

    output_path = output_dir / 'experiment_design.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"✓ Saved: {output_path}")


if __name__ == '__main__':
    main()
