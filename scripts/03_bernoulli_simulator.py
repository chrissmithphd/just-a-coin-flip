#!/usr/bin/env python3
"""
Bernoulli NFL Simulator

Generate synthetic NFL histories by replacing actual game outcomes with
Bernoulli(p_i) draws, keeping all other game attributes (date, teams, probabilities)
identical to the real NFL.

This creates matched counterfactual histories where games are purely random
conditional on their pregame market probabilities.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


class BernoulliNFL:
    """
    Simulate NFL game outcomes as independent Bernoulli draws.

    Each game i has a vig-free home win probability p_i from the closing line.
    We generate synthetic outcomes Y_i* ~ Bernoulli(p_i) while preserving
    all game metadata (date, teams, season, probabilities).
    """

    def __init__(self, games_data: List[Dict[str, Any]], seed: int = 42):
        """
        Initialize simulator with real NFL game data.

        Args:
            games_data: List of processed game dicts with p_home_vig_free
            seed: Random seed for reproducibility
        """
        self.games_real = games_data
        self.n_games = len(games_data)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Extract probabilities for vectorized simulation
        self.p_home = np.array([g['p_home_vig_free'] for g in games_data])

        # Cache real outcomes for comparison
        self.y_real = np.array([int(g['home_win']) for g in games_data])

    def simulate_season(self, seed: int = None) -> np.ndarray:
        """
        Generate one complete synthetic NFL history.

        Args:
            seed: Optional seed override for this simulation

        Returns:
            Array of shape (n_games,) with 1 = home win, 0 = away win
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng

        # Draw from Bernoulli(p_i) for each game
        return (rng.random(self.n_games) < self.p_home).astype(int)

    def simulate_monte_carlo(self, n_sims: int = 10000, verbose: bool = True) -> np.ndarray:
        """
        Generate many synthetic NFL histories for Monte Carlo inference.

        Args:
            n_sims: Number of complete NFL histories to simulate
            verbose: Print progress updates

        Returns:
            Array of shape (n_sims, n_games) with synthetic outcomes
        """
        if verbose:
            print(f"Simulating {n_sims:,} synthetic NFL histories...")

        # Use different seeds for each simulation
        seeds = self.rng.randint(0, 2**31, size=n_sims)

        outcomes = np.zeros((n_sims, self.n_games), dtype=np.int8)

        for i, seed in enumerate(seeds):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  {i+1:,} / {n_sims:,} complete")
            outcomes[i] = self.simulate_season(seed=seed)

        if verbose:
            print(f"✓ Monte Carlo simulation complete")

        return outcomes

    def get_synthetic_games(self, y_synthetic: np.ndarray) -> List[Dict[str, Any]]:
        """
        Create full game records with synthetic outcomes.

        Args:
            y_synthetic: Array of synthetic home win indicators (0/1)

        Returns:
            List of game dicts with synthetic outcomes replacing real ones
        """
        synthetic_games = []

        for i, game_real in enumerate(self.games_real):
            game_synth = game_real.copy()

            # Replace outcome
            home_win_synth = bool(y_synthetic[i])
            game_synth['home_win'] = home_win_synth
            game_synth['synthetic'] = True

            # Scores become meaningless in synthetic world, mark as NA
            game_synth['home_score'] = None
            game_synth['away_score'] = None

            synthetic_games.append(game_synth)

        return synthetic_games


def load_and_simulate(data_path: str, n_sims: int = 10000, seed: int = 42):
    """
    Load real NFL data and generate Monte Carlo simulations.

    Args:
        data_path: Path to processed games JSON
        n_sims: Number of synthetic histories
        seed: Random seed

    Returns:
        Tuple of (real_games, simulator, monte_carlo_outcomes)
    """
    # Load real data
    print(f"Loading real NFL data from {data_path}...")
    with open(data_path) as f:
        games_real = json.load(f)
    print(f"Loaded {len(games_real):,} games")

    # Initialize simulator
    print(f"Initializing Bernoulli simulator (seed={seed})...")
    simulator = BernoulliNFL(games_real, seed=seed)

    # Run Monte Carlo
    mc_outcomes = simulator.simulate_monte_carlo(n_sims=n_sims)

    return games_real, simulator, mc_outcomes


def save_monte_carlo(mc_outcomes: np.ndarray, output_path: str):
    """
    Save Monte Carlo outcomes to compressed numpy format.

    Args:
        mc_outcomes: Array of shape (n_sims, n_games)
        output_path: Path to save .npz file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_path, outcomes=mc_outcomes)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved Monte Carlo outcomes: {output_path} ({file_size_mb:.1f} MB)")


def main():
    """Generate and save Monte Carlo simulations."""

    # Configuration
    DATA_PATH = 'data/processed/nfl_games_processed.json'
    OUTPUT_PATH = 'data/simulated/monte_carlo_outcomes.npz'
    N_SIMS = 10000
    SEED = 42

    # Load and simulate
    games_real, simulator, mc_outcomes = load_and_simulate(
        DATA_PATH, n_sims=N_SIMS, seed=SEED
    )

    # Summary statistics
    print("\n" + "=" * 70)
    print("MONTE CARLO SUMMARY")
    print("=" * 70)
    print(f"\nReal NFL:")
    print(f"  Total games: {len(games_real):,}")
    print(f"  Home wins: {simulator.y_real.sum():,} ({simulator.y_real.mean():.3f})")
    print(f"  Average p_home: {simulator.p_home.mean():.3f}")

    print(f"\nSynthetic NFL (Monte Carlo):")
    print(f"  Simulations: {N_SIMS:,}")
    print(f"  Average home wins: {mc_outcomes.sum(axis=1).mean():.1f} ± {mc_outcomes.sum(axis=1).std():.1f}")
    print(f"  Average home win rate: {mc_outcomes.mean():.3f}")
    print(f"  Expected (= avg p_home): {simulator.p_home.mean():.3f}")

    # Check if real NFL is an outlier
    real_total_wins = simulator.y_real.sum()
    mc_total_wins = mc_outcomes.sum(axis=1)

    percentile_rank = (mc_total_wins < real_total_wins).mean() * 100

    print(f"\nReal NFL total home wins: {real_total_wins}")
    print(f"  Percentile rank in MC distribution: {percentile_rank:.1f}%")

    if percentile_rank < 2.5 or percentile_rank > 97.5:
        print(f"  ⚠️  Real NFL is in outer 5% of Bernoulli distribution!")
    else:
        print(f"  ✓ Real NFL consistent with Bernoulli model (total home wins)")

    print("=" * 70 + "\n")

    # Save
    save_monte_carlo(mc_outcomes, OUTPUT_PATH)

    print(f"✓ Simulation complete!")


if __name__ == '__main__':
    main()
