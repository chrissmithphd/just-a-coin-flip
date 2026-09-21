# NFL Game Outcomes as Bernoulli Draws

A statistical research project testing whether professional sports game outcomes behave like conditionally independent Bernoulli random variables when conditioned on pregame market-implied win probabilities.

## Research Question

**Do NFL game outcomes follow a pure random-game model once we condition on market probabilities?**

For game *i*, let *p_i* be the vig-adjusted pregame market probability that the home team wins.

The theoretical random-game model is:
```
Y_i* ~ Bernoulli(p_i)
```

where *Y_i** is a synthetic game outcome.

The real outcome is:
```
Y_i = 1 if home team won, 0 otherwise
```

We test whether the time-series and team-level behavior of real outcomes *Y_i* can be distinguished from synthetic Bernoulli outcomes generated using exactly the same *p_i* probabilities.

**Key insight:** This does *not* test whether markets are predictive (they clearly are). Instead, it tests whether games are *random conditional on the market price* — i.e., whether the stochastic process generating outcomes matches the theoretical Bernoulli model, or exhibits autocorrelation, momentum, mean reversion, or team-level clustering that violates the independence assumption.

## Dataset

**Source:** [Sportsbook Review Historical Data](https://github.com/flancast90/sportsbookreview-scraper)  
**Coverage:** 2011-2021 NFL seasons  
**Games:** 2,946 (regular season + playoffs, excluding Super Bowls)  
**Completeness:** 0% missing moneyline data

### Data Fields

Each game record includes:
- Game identifiers (date, teams, season)
- Final scores
- **Closing moneyline odds** (home and away)
- Vig-adjusted win probabilities

### Key Statistics

| Metric | Value |
|--------|-------|
| Home win rate (actual) | 55.8% |
| Home win probability (market avg) | 56.6% |
| Calibration gap | -1.4% |
| Average sportsbook vig | 3.83% |

## Methodology

### Phase 1: Data Discovery ✅

1. Identified and downloaded historical NFL moneyline odds
2. Converted American odds to implied probabilities
3. Removed sportsbook vig (normalized probabilities to sum to 1)
4. Validated data quality and coverage
5. Generated exploratory visualizations

### Phase 2: Bernoulli Independence Tests (Planned)

Test statistics to distinguish real outcomes from synthetic Bernoulli draws:

1. **Runs tests** — Are streaks (WWLWW) longer or shorter than expected?
2. **Autocorrelation** — Do recent outcomes predict future outcomes conditional on *p_i*?
3. **Team-level clustering** — Do outcomes exhibit team-specific patterns beyond market expectations?
4. **Calendar effects** — Do outcomes depend on day of week, bye weeks, or schedule position?
5. **Conditional variance** — Does the variance of outcomes match Bernoulli(*p_i*)?

Each test compares the observed statistic against the distribution obtained from 10,000 Monte Carlo simulations of the Bernoulli model.

## Project Structure

```
.
├── data/
│   ├── raw/                    # Raw JSON from Sportsbook Review
│   └── processed/              # Clean data with vig-free probabilities
├── scripts/
│   ├── 01_process_raw_data.py  # Odds → probabilities conversion
│   └── 02_exploratory_analysis.py  # Summary stats and plots
├── output/
│   └── exploratory/            # Plots and visualizations
├── docs/
│   └── DATA_DISCOVERY.md       # Data source evaluation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/chrissmithphd/just-a-coin-flip.git
cd just-a-coin-flip

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Process Raw Data

Convert raw moneyline odds to vig-free probabilities:

```bash
python scripts/01_process_raw_data.py
```

**Inputs:** `data/raw/nfl_sbr_10Y.json`  
**Outputs:** `data/processed/nfl_games_processed.json`

### Exploratory Analysis

Generate summary statistics and visualizations:

```bash
python scripts/02_exploratory_analysis.py
```

**Outputs:**
- `output/exploratory/games_by_season.png` — Games per season
- `output/exploratory/win_prob_distribution.png` — Distribution of market probabilities
- `output/exploratory/calibration_curve.png` — Predicted vs actual win rates
- `output/exploratory/home_win_rate.png` — Overall home field advantage
- `output/exploratory/missing_odds.png` — Data completeness check

## Results (Preliminary)

### Market Calibration

The closing moneyline odds are well-calibrated:
- **Expected home wins** (sum of *p_i*): 1,666.7
- **Actual home wins**: 1,644
- **Difference**: -22.7 games (-1.4%)

This suggests markets are approximately unbiased over the 11-year period.

### Home Field Advantage

Markets consistently price in home field advantage:
- **Average home win probability**: 56.6%
- **Actual home win rate**: 55.8%

Both market and outcomes show clear home advantage vs. the 50% baseline of a neutral-site fair coin.

## Interpretation

**This is NOT a predictive model.** We do not attempt to forecast game outcomes or beat the market.

Instead, we test a fundamental assumption in sports analytics: that once you condition on all available information (proxied by the market price), the remaining uncertainty is purely random.

**If true:** Games are "fair coins" with bias *p_i*, and betting markets are informationally efficient.

**If false:** Outcomes exhibit patterns (momentum, fatigue, schedule effects, referee bias, etc.) that persist even after conditioning on market prices, suggesting either:
1. Markets systematically misprice certain situations, or
2. True randomness is violated (e.g., by hidden information or strategic play)

## Future Work

- [ ] Extend dataset to 2022-2024 (via The Odds API or updated scrapes)
- [ ] Implement Bernoulli simulation engine
- [ ] Run full test battery (runs, autocorrelation, team clustering)
- [ ] Conditional analysis by:
  - Home favorite vs underdog
  - Divisional vs conference vs inter-conference games
  - Spread size bins
  - Time of season
- [ ] Cross-sport comparison (NBA, MLB, NHL)

## References

- Sportsbook Review historical odds: [flancast90/sportsbookreview-scraper](https://github.com/flancast90/sportsbookreview-scraper)
- Efficient markets hypothesis in sports betting: Thaler & Ziemba (1988)
- Market-based probabilities: Wolfers & Zitzewitz (2004)

## License

MIT License - see LICENSE file for details.

## Contact

**Chris Smith** — [@chrissmithphd](https://github.com/chrissmithphd)

Project Link: [https://github.com/chrissmithphd/just-a-coin-flip](https://github.com/chrissmithphd/just-a-coin-flip)

---

*Generated with Claude Code*
