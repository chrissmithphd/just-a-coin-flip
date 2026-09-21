# NFL Betting Data Discovery Report

## Date: 2026-09-21

## Research Question

Testing whether NFL game outcomes behave like conditionally independent Bernoulli draws when conditioned on pregame market-implied win probabilities.

## Data Requirements

For each NFL game, we need:
1. Unique game identifier
2. Game date and time
3. Season
4. Home team
5. Away team  
6. Final scores (home and away)
7. **Pregame moneyline odds** (home and away)
8. Sportsbook source
9. Timestamp of odds (ideally closing line or pre-kickoff snapshot)

## Dataset Found: Sportsbook Review Historical Data

### Source
- **Repository**: [flancast90/sportsbookreview-scraper](https://github.com/flancast90/sportsbookreview-scraper)
- **License**: MIT
- **Last Updated**: 2026-09-18
- **Stars**: 53

### Coverage
- **Years**: 2011-2021 (11 seasons)
- **Total NFL Games**: 2,956
- **Games per season**: 267-284 (consistent with NFL regular + playoff schedule)
- **Missing moneyline data**: 0% (complete coverage)

### Data Fields Available
```json
{
  "season": 2011,
  "date": 20110908.0,
  "home_team": "Packers",
  "away_team": "Saints",
  "home_1stQtr": "21",
  "away_1stQtr": "7",
  "home_2ndQtr": "7",
  "away_2ndQtr": "10",
  "home_3rdQtr": "7",
  "away_3rdQtr": "10",
  "home_4thQtr": "7",
  "away_4thQtr": "7",
  "home_final": "42",
  "away_final": "34",
  "home_close_ml": -250,
  "away_close_ml": 210,
  "home_open_spread": -4.5,
  "away_open_spread": 4.5,
  "home_close_spread": -5.0,
  "away_close_spread": 5.0,
  "home_2H_spread": -24.5,
  "away_2H_spread": 24.5,
  "2H_total": 0.0,
  "open_over_under": 46.0,
  "close_over_under": 48.0
}
```

### Key Features
- **Closing moneylines**: `home_close_ml` and `away_close_ml` represent the closing pregame odds
- **Complete scoring data**: Quarter-by-quarter and final scores
- **Spreads and totals**: Additional betting lines available (not needed for core analysis but useful for validation)

### Moneyline Range
- Home team: -5000 to +1100
- Away team: -2500 to +2173

### Data Quality Assessment

**Strengths:**
1. ✅ Zero missing moneyline data
2. ✅ Consistent seasonal coverage (267+ games/season)
3. ✅ Uses closing lines (not stale early-week odds)
4. ✅ Complete game results
5. ✅ 11 years of data provides substantial sample size

**Limitations:**
1. ⚠️ Data ends at 2021 (need 2022-2024 for full currency)
2. ⚠️ Sportsbook source not explicitly identified (likely SBR consensus line)
3. ⚠️ Exact timestamp of odds snapshot not provided (assumed to be closing line)
4. ⚠️ Team names may need standardization
5. ⚠️ Date format is YYYYMMDD as float, needs parsing

**Critical Validation Needed:**
- Confirm odds were captured pre-kickoff (not retrospectively)
- Verify no information leakage (odds recorded after game started)
- Check for definitional consistency of "closing line"

## Alternative/Complementary Sources Investigated

### 1. nflverse/nflverse-data
- **Status**: No betting odds data found
- **Coverage**: Comprehensive play-by-play, rosters, stats (2002+)
- **Use case**: Could supplement with additional game context, but lacks odds

### 2. The Odds API
- **Status**: Paid service only ($30-249/month)
- **Coverage**: Historical data back to 2020
- **Use case**: Could extend 2022-2024, but requires paid subscription

### 3. Kaggle Datasets
- **Status**: Several NFL betting datasets exist but specific accessibility unclear
- **Notable**: "NFL scores and betting data" by Toby Crabtree appears related to SBR scraper

### 4. Pro Football Reference
- **Status**: 403 Forbidden on web scraping
- **Coverage**: Comprehensive game data but betting odds not prominently advertised

## Recommendation

**Use the SBR 2011-2021 dataset as the primary data source for Phase 1 analysis.**

**Rationale:**
1. Complete moneyline coverage with zero missing data
2. Closing lines minimize timing issues
3. 2,956 games provides strong statistical power
4. Clean, structured JSON format
5. Free and immediately accessible

**Next Steps:**
1. Load and validate the SBR dataset
2. Convert American odds to probabilities
3. Apply vig removal (normalize to sum=1)
4. Create derived fields:
   - `game_id`
   - `game_datetime` (from date field)
   - `home_win` (boolean)
   - `p_home_vig_free` (vig-adjusted probability)
5. Generate exploratory visualizations:
   - Games by season
   - Distribution of win probabilities
   - Actual home win rate overall and by probability bin
   - Calibration curves

**For Extended Coverage (2022-2024):**
- Investigate paid access to The Odds API
- Check if SBR scraper can be run for recent seasons
- Look for community-maintained updated datasets

## Data Integrity Concerns to Address

1. **Reverse Causality**: Confirm odds timestamp is truly pregame
2. **Line Shopping**: SBR likely uses a consensus line, but which books?
3. **Steam Moves**: Late line moves from sharp money could affect interpretation
4. **Vig Calculation**: Different books have different vigs; need to verify normalization approach
5. **Team Name Mapping**: Standardize team names for any merges or extensions

## Dataset Location

```
/home/cbsmith/git/stats/data/raw/nfl_sbr_10Y.json
```

**File Size**: 1.5 MB
**Format**: JSON array of game objects
**Records**: 2,956 games
