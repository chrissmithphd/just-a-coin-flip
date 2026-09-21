#!/usr/bin/env python3
"""
Process raw NFL betting data from Sportsbook Review into analysis-ready format.

Converts American moneyline odds to vig-free probabilities and creates
a clean dataset for Bernoulli independence testing.
"""

import json
from datetime import datetime
from pathlib import Path


def american_to_prob(moneyline):
    """
    Convert American moneyline odds to implied probability.

    Positive odds (e.g., +150): prob = 100 / (odds + 100)
    Negative odds (e.g., -200): prob = |odds| / (|odds| + 100)

    Args:
        moneyline: American odds (e.g., -150, +200)

    Returns:
        Implied probability between 0 and 1
    """
    if moneyline > 0:
        return 100 / (moneyline + 100)
    else:
        return abs(moneyline) / (abs(moneyline) + 100)


def remove_vig(prob_home, prob_away):
    """
    Remove sportsbook vig by normalizing probabilities to sum to 1.

    The "vig" or "juice" is the sportsbook's margin. Raw probabilities
    from odds typically sum to > 1. We normalize so they sum exactly to 1,
    effectively removing the vig and recovering the market's true belief.

    Args:
        prob_home: Raw implied probability for home team
        prob_away: Raw implied probability for away team

    Returns:
        Tuple of (vig-free home prob, vig-free away prob)
    """
    total = prob_home + prob_away
    return prob_home / total, prob_away / total


def parse_date(date_float):
    """
    Parse YYYYMMDD float to datetime object.

    Args:
        date_float: Date as float (e.g., 20110908.0)

    Returns:
        datetime object
    """
    date_str = str(int(date_float))
    return datetime.strptime(date_str, "%Y%m%d")


def process_game(game_raw, idx):
    """
    Transform raw game dict into analysis format.

    Args:
        game_raw: Raw game dict from SBR data
        idx: Game index for creating unique ID

    Returns:
        Processed game dict with derived fields
    """
    # Parse basic fields
    season = game_raw['season']
    game_date = parse_date(game_raw['date'])
    home_team = game_raw['home_team']
    away_team = game_raw['away_team']

    # Parse scores
    home_score = int(game_raw['home_final'])
    away_score = int(game_raw['away_final'])
    home_win = home_score > away_score

    # Parse moneylines
    home_ml = game_raw['home_close_ml']
    away_ml = game_raw['away_close_ml']

    # Convert to probabilities
    p_home_raw = american_to_prob(home_ml)
    p_away_raw = american_to_prob(away_ml)

    # Remove vig
    p_home_vig_free, p_away_vig_free = remove_vig(p_home_raw, p_away_raw)

    # Create unique game ID
    game_id = f"{season}_{game_date.strftime('%Y%m%d')}_{away_team}_{home_team}"

    return {
        'game_id': game_id,
        'idx': idx,
        'season': season,
        'game_date': game_date.isoformat(),
        'game_datetime': game_date.isoformat(),  # No time data available, date only
        'home_team': home_team,
        'away_team': away_team,
        'home_score': home_score,
        'away_score': away_score,
        'home_win': home_win,
        'home_moneyline': home_ml,
        'away_moneyline': away_ml,
        'p_home_raw': round(p_home_raw, 6),
        'p_away_raw': round(p_away_raw, 6),
        'p_home_vig_free': round(p_home_vig_free, 6),
        'p_away_vig_free': round(p_away_vig_free, 6),
        'vig': round((p_home_raw + p_away_raw - 1) * 100, 2),  # Vig as percentage
    }


def main():
    """Load, process, and save NFL betting data."""

    # Paths
    raw_path = Path('data/raw/nfl_sbr_10Y.json')
    processed_path = Path('data/processed/nfl_games_processed.json')
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"Loading raw data from {raw_path}...")
    with open(raw_path) as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} games")

    # Filter out Super Bowls (neutral site, no home team)
    # These have home_team = "0" or home_close_ml = 0
    regular_games = [
        g for g in raw_data
        if g['home_team'] != "0" and g['home_close_ml'] != 0 and g['away_close_ml'] != 0
    ]
    super_bowls = len(raw_data) - len(regular_games)
    if super_bowls > 0:
        print(f"Filtered {super_bowls} Super Bowl games (neutral site)")
    print(f"Processing {len(regular_games)} regular/playoff games with home teams...")

    # Process games
    processed_games = [process_game(game, idx) for idx, game in enumerate(regular_games)]

    # Basic validation
    print("\nValidation:")
    print(f"  Total games processed: {len(processed_games)}")
    print(f"  Seasons: {min(g['season'] for g in processed_games)} - {max(g['season'] for g in processed_games)}")

    # Check probability bounds
    invalid_probs = [
        g for g in processed_games
        if not (0 < g['p_home_vig_free'] < 1 and 0 < g['p_away_vig_free'] < 1)
    ]
    if invalid_probs:
        print(f"  WARNING: {len(invalid_probs)} games with invalid probabilities!")
    else:
        print(f"  ✓ All probabilities in valid range (0, 1)")

    # Check probability sums
    sum_errors = [
        abs(g['p_home_vig_free'] + g['p_away_vig_free'] - 1.0)
        for g in processed_games
    ]
    max_sum_error = max(sum_errors)
    if max_sum_error > 1e-6:
        print(f"  WARNING: Max probability sum error: {max_sum_error:.2e}")
    else:
        print(f"  ✓ Probabilities sum to 1.0 (max error: {max_sum_error:.2e})")

    # Summary statistics
    avg_vig = sum(g['vig'] for g in processed_games) / len(processed_games)
    avg_p_home = sum(g['p_home_vig_free'] for g in processed_games) / len(processed_games)
    actual_home_wins = sum(g['home_win'] for g in processed_games)
    home_win_rate = actual_home_wins / len(processed_games)

    print(f"\nSummary Statistics:")
    print(f"  Average vig: {avg_vig:.2f}%")
    print(f"  Average home win probability (vig-free): {avg_p_home:.3f}")
    print(f"  Actual home wins: {actual_home_wins} / {len(processed_games)} = {home_win_rate:.3f}")

    # Save processed data
    print(f"\nSaving processed data to {processed_path}...")
    with open(processed_path, 'w') as f:
        json.dump(processed_games, f, indent=2)

    print("✓ Processing complete!")

    # Show examples
    print("\nExample processed records:")
    for game in processed_games[:3]:
        print(f"\n{game['game_id']}:")
        print(f"  {game['away_team']} @ {game['home_team']}")
        print(f"  Score: {game['away_score']}-{game['home_score']} (home {'win' if game['home_win'] else 'loss'})")
        print(f"  Moneylines: {game['away_moneyline']}/{game['home_moneyline']}")
        print(f"  P(home): {game['p_home_vig_free']:.3f} (vig: {game['vig']:.2f}%)")


if __name__ == '__main__':
    main()
