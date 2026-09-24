#!/usr/bin/env python3
"""
Market vs History: Predictive Test

Does recent team performance contain information beyond what the
betting market already knows?

Test whether adding recent history features improves out-of-sample
prediction beyond the market probability alone.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    from sklearn.linear_model import LogisticRegression
    HAS_STATSMODELS = False


def load_data():
    """Load real games and Monte Carlo simulations."""
    with open('data/processed/nfl_games_processed.json') as f:
        games = json.load(f)

    mc_data = np.load('data/simulated/monte_carlo_outcomes.npz')
    mc_outcomes = mc_data['outcomes']

    return games, mc_outcomes


def build_team_histories(games, outcomes):
    """
    Build chronological game sequences for each team with outcomes.

    Args:
        games: List of game dicts
        outcomes: Array of home win indicators (n_games,) or (n_sims, n_games)

    Returns:
        Dict mapping team to list of (game_idx, won, p_win, game_date)
    """
    is_mc = len(outcomes.shape) == 2
    team_histories = defaultdict(list)

    for i, game in enumerate(games):
        game_date = game['game_datetime']

        # Home team
        home_team = game['home_team']
        p_home = game['p_home_vig_free']

        if is_mc:
            home_outcomes = outcomes[:, i]  # shape (n_sims,)
            team_histories[home_team].append({
                'game_idx': i,
                'date': game_date,
                'won': home_outcomes,
                'p_win': p_home,
                'is_home': True
            })
        else:
            home_won = int(outcomes[i])
            team_histories[home_team].append({
                'game_idx': i,
                'date': game_date,
                'won': home_won,
                'p_win': p_home,
                'is_home': True
            })

        # Away team
        away_team = game['away_team']
        p_away = 1 - p_home

        if is_mc:
            away_outcomes = 1 - outcomes[:, i]
            team_histories[away_team].append({
                'game_idx': i,
                'date': game_date,
                'won': away_outcomes,
                'p_win': p_away,
                'is_home': False
            })
        else:
            away_won = int(1 - outcomes[i])
            team_histories[away_team].append({
                'game_idx': i,
                'date': game_date,
                'won': away_won,
                'p_win': p_away,
                'is_home': False
            })

    # Sort each team's history by date
    for team in team_histories:
        team_histories[team] = sorted(team_histories[team], key=lambda x: x['date'])

    return team_histories


def compute_team_features(team_history, game_idx, sim_idx=None):
    """
    Compute recent-history features for a team at a specific game.
    Uses only games BEFORE game_idx.

    Args:
        team_history: List of team's games (sorted chronologically)
        game_idx: Index of current game (to predict)
        sim_idx: Simulation index (for MC data), None for real data

    Returns:
        Dict of features, or None if insufficient history
    """
    # Find position of current game in team's history
    current_pos = None
    for pos, game_info in enumerate(team_history):
        if game_info['game_idx'] == game_idx:
            current_pos = pos
            break

    if current_pos is None or current_pos == 0:
        return None  # No prior games

    # Collect prior games
    prior_games = team_history[:current_pos]

    if len(prior_games) == 0:
        return None

    # Extract outcomes and probabilities
    if sim_idx is not None:
        # MC data: outcomes are arrays
        prior_outcomes = np.array([g['won'][sim_idx] for g in prior_games])
        prior_probs = np.array([g['p_win'] for g in prior_games])
    else:
        # Real data
        prior_outcomes = np.array([g['won'] for g in prior_games])
        prior_probs = np.array([g['p_win'] for g in prior_games])

    # Compute residuals
    residuals = prior_outcomes - prior_probs

    # Features
    features = {}

    # Last game residual
    features['prev_1_residual'] = residuals[-1]

    # Rolling averages
    if len(residuals) >= 3:
        features['prev_3_residual'] = residuals[-3:].mean()
    else:
        features['prev_3_residual'] = residuals.mean()

    if len(residuals) >= 5:
        features['prev_5_residual'] = residuals[-5:].mean()
    else:
        features['prev_5_residual'] = residuals.mean()

    # Previous game outcome
    features['prev_win'] = int(prior_outcomes[-1])

    # Current streak length (signed: positive for wins, negative for losses)
    streak = 0
    for outcome in reversed(prior_outcomes):
        if outcome == prior_outcomes[-1]:
            streak += 1
        else:
            break
    features['streak_length'] = streak if prior_outcomes[-1] == 1 else -streak

    return features


def build_dataset(games, outcomes, team_histories):
    """
    Build dataset with features for each game.

    Returns:
        List of dicts with {game_idx, y, p_market, features}
    """
    is_mc = len(outcomes.shape) == 2
    n_sims = outcomes.shape[0] if is_mc else 1

    dataset = []

    for game_idx, game in enumerate(games):
        home_team = game['home_team']
        away_team = game['away_team']
        p_home = game['p_home_vig_free']

        if is_mc:
            y_home = outcomes[:, game_idx]  # shape (n_sims,)
        else:
            y_home = outcomes[game_idx]

        # Get features for home team
        for sim_idx in range(n_sims) if is_mc else [None]:
            features_home = compute_team_features(
                team_histories[home_team], game_idx, sim_idx
            )

            if features_home is not None:
                record = {
                    'game_idx': game_idx,
                    'team': home_team,
                    'y': y_home[sim_idx] if is_mc else y_home,
                    'p_market': p_home,
                    'features': features_home,
                    'sim_idx': sim_idx
                }
                dataset.append(record)

        # Get features for away team
        for sim_idx in range(n_sims) if is_mc else [None]:
            features_away = compute_team_features(
                team_histories[away_team], game_idx, sim_idx
            )

            if features_away is not None:
                record = {
                    'game_idx': game_idx,
                    'team': away_team,
                    'y': (1 - y_home[sim_idx]) if is_mc else (1 - y_home),
                    'p_market': 1 - p_home,
                    'features': features_away,
                    'sim_idx': sim_idx
                }
                dataset.append(record)

    return dataset


def logit(p):
    """Logit transform with clipping."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def inv_logit(x):
    """Inverse logit."""
    return 1 / (1 + np.exp(-x))


def fit_model_with_offset(y, X, offset, method='statsmodels'):
    """
    Fit logistic regression with offset.

    Returns:
        Fitted model (callable that takes X, offset and returns predictions)
    """
    if method == 'statsmodels' and HAS_STATSMODELS:
        model = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset)
        result = model.fit(disp=0)

        def predict_fn(X_new, offset_new):
            linear = result.predict(X_new) + offset_new
            return inv_logit(linear)

        return predict_fn, result.params

    else:
        # Fallback: manual implementation
        # logit(p) = offset + X @ beta
        # We fit beta by logistic regression on (y, X) with offset absorbed

        # Transform: new_y = logit(p) - offset
        # Approximate by fitting X to predict (y - inv_logit(offset))

        # Simpler: fit without offset, predictions = inv_logit(logit(p_market) + X @ beta)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X, y)

        def predict_fn(X_new, offset_new):
            logit_pred = model.predict_log_proba(X_new)[:, 1] + offset_new
            return inv_logit(logit_pred)

        return predict_fn, model.coef_[0]


def expanding_walk_forward_cv(dataset, feature_sets, n_folds=5):
    """
    Expanding window walk-forward validation.

    For each fold:
    - Train on games 0 to split_idx
    - Test on games split_idx to split_idx + fold_size
    - Expand training set for next fold

    Args:
        dataset: List of records with features
        feature_sets: Dict mapping name to list of feature keys
        n_folds: Number of folds

    Returns:
        Dict mapping feature_set_name to out-of-sample predictions
    """
    # Sort by game index
    dataset = sorted(dataset, key=lambda x: x['game_idx'])

    n = len(dataset)
    fold_size = n // (n_folds + 1)  # Reserve first portion for initial training

    results = {name: {'y_true': [], 'y_pred': []} for name in feature_sets}

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > n:
            test_end = n

        train_data = dataset[:train_end]
        test_data = dataset[test_start:test_end]

        if len(test_data) == 0:
            break

        # Prepare train data
        y_train = np.array([d['y'] for d in train_data])
        p_market_train = np.array([d['p_market'] for d in train_data])
        offset_train = logit(p_market_train)

        # Prepare test data
        y_test = np.array([d['y'] for d in test_data])
        p_market_test = np.array([d['p_market'] for d in test_data])
        offset_test = logit(p_market_test)

        # Fit and evaluate each feature set
        for name, feature_keys in feature_sets.items():
            if name == 'market_only':
                # Just use market probability
                preds = p_market_test
            else:
                # Build feature matrix
                X_train = np.column_stack([
                    [d['features'][k] for d in train_data]
                    for k in feature_keys
                ])
                X_test = np.column_stack([
                    [d['features'][k] for d in test_data]
                    for k in feature_keys
                ])

                # Fit model with offset
                predict_fn, _ = fit_model_with_offset(
                    y_train, X_train, offset_train,
                    method='statsmodels' if HAS_STATSMODELS else 'sklearn'
                )

                # Predict
                preds = predict_fn(X_test, offset_test)

            results[name]['y_true'].extend(y_test)
            results[name]['y_pred'].extend(preds)

    return results


def compute_metrics(y_true, y_pred):
    """Compute log loss and Brier score."""
    y_true = np.array(y_true)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

    # Log loss
    log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Brier score
    brier = np.mean((y_true - y_pred) ** 2)

    return {'log_loss': log_loss, 'brier': brier}


def run_experiment(games, outcomes, n_mc_sims=1000):
    """
    Run the full experiment on real data and MC simulations.

    Returns:
        Dict with results for real and MC
    """
    print(f"\n{'='*70}")
    print("MARKET VS HISTORY EXPERIMENT")
    print(f"{'='*70}\n")

    # Define feature sets
    feature_sets = {
        'market_only': [],
        'plus_last_1': ['prev_1_residual'],
        'plus_last_3': ['prev_3_residual'],
        'plus_last_5': ['prev_5_residual'],
        'plus_all': ['prev_1_residual', 'prev_3_residual', 'prev_5_residual',
                     'prev_win', 'streak_length']
    }

    # Real NFL
    print("Processing real NFL data...")
    y_real = np.array([int(g['home_win']) for g in games])
    team_histories_real = build_team_histories(games, y_real)
    dataset_real = build_dataset(games, y_real, team_histories_real)
    print(f"  Dataset: {len(dataset_real)} team-games with sufficient history")

    print("\nRunning expanding walk-forward cross-validation...")
    results_real = expanding_walk_forward_cv(dataset_real, feature_sets, n_folds=5)

    metrics_real = {}
    for name in feature_sets:
        metrics_real[name] = compute_metrics(
            results_real[name]['y_true'],
            results_real[name]['y_pred']
        )

    print("\nReal NFL out-of-sample performance:")
    baseline_ll = metrics_real['market_only']['log_loss']
    for name in feature_sets:
        ll = metrics_real[name]['log_loss']
        delta = ll - baseline_ll
        print(f"  {name:20s}: log_loss={ll:.4f}  (Δ={delta:+.4f})")

    # Monte Carlo simulations
    print(f"\n\nRunning {n_mc_sims} Monte Carlo simulations...")

    mc_data = np.load('data/simulated/monte_carlo_outcomes.npz')
    mc_outcomes = mc_data['outcomes'][:n_mc_sims]  # Use first n_mc_sims

    mc_deltas = {name: [] for name in feature_sets if name != 'market_only'}

    for sim_idx in range(n_mc_sims):
        if (sim_idx + 1) % 100 == 0:
            print(f"  Simulation {sim_idx + 1}/{n_mc_sims}")

        # Build dataset for this simulation
        outcomes_sim = mc_outcomes[sim_idx]
        team_histories_sim = build_team_histories(games, outcomes_sim)
        dataset_sim = build_dataset(games, outcomes_sim, team_histories_sim)

        # Run CV
        results_sim = expanding_walk_forward_cv(dataset_sim, feature_sets, n_folds=5)

        # Compute metrics
        metrics_sim = {}
        for name in feature_sets:
            metrics_sim[name] = compute_metrics(
                results_sim[name]['y_true'],
                results_sim[name]['y_pred']
            )

        # Store deltas
        baseline_ll_sim = metrics_sim['market_only']['log_loss']
        for name in mc_deltas:
            delta = metrics_sim[name]['log_loss'] - baseline_ll_sim
            mc_deltas[name].append(delta)

    print("\n✓ Monte Carlo complete")

    # Compile results
    results = {
        'real': {
            'metrics': metrics_real,
            'baseline_ll': baseline_ll,
            'deltas': {name: metrics_real[name]['log_loss'] - baseline_ll
                      for name in feature_sets if name != 'market_only'}
        },
        'mc': {
            'deltas': mc_deltas,
            'percentiles': {}
        }
    }

    # Compute percentiles
    for name in mc_deltas:
        mc_vals = np.array(mc_deltas[name])
        real_delta = results['real']['deltas'][name]
        percentile = (mc_vals < real_delta).mean() * 100

        results['mc']['percentiles'][name] = {
            'real_delta': real_delta,
            'mc_mean': mc_vals.mean(),
            'mc_std': mc_vals.std(),
            'percentile': percentile,
            'ci_lower': np.percentile(mc_vals, 2.5),
            'ci_upper': np.percentile(mc_vals, 97.5)
        }

    return results


def main():
    """Run market vs history experiment."""
    output_dir = Path('output/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    games, mc_outcomes = load_data()

    # Run experiment with 1000 MC sims for development
    results = run_experiment(games, mc_outcomes, n_mc_sims=1000)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    for name, stats in results['mc']['percentiles'].items():
        print(f"{name}:")
        print(f"  Real Δ log loss: {stats['real_delta']:+.4f}")
        print(f"  MC mean:         {stats['mc_mean']:+.4f} ± {stats['mc_std']:.4f}")
        print(f"  95% CI:          [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}]")
        print(f"  Percentile:      {stats['percentile']:.1f}%")
        print()

    # Save results
    results_path = output_dir / 'market_vs_history_results.json'

    # Convert numpy arrays to lists for JSON
    results_to_save = {
        'real': results['real'],
        'mc': {
            'percentiles': results['mc']['percentiles'],
            'deltas_summary': {
                name: {
                    'mean': float(np.mean(deltas)),
                    'std': float(np.std(deltas)),
                    'min': float(np.min(deltas)),
                    'max': float(np.max(deltas))
                }
                for name, deltas in results['mc']['deltas'].items()
            }
        }
    }

    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"Saved results: {results_path}")
    print("\n✓ Experiment complete!")


if __name__ == '__main__':
    main()
