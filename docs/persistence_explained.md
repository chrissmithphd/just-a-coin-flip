# Streaks Are Real — But the Market Already Knows

This note explains the one place NFL results *look* like they beat a coin flip,
why that appearance is misleading, and how the correct test resolves it. Every
number is produced by `scripts/06_analysis_team_timeseries.py`.

---

## The finding in one sentence

NFL teams are genuinely streakier than independent coin flips, but the streakiness
is entirely explained by the betting line moving after each game — so there is no
predictable edge left once you condition on the market probability.

---

## Step 1: The streaks are real

Compared with 10,000 simulated leagues of independent coin flips weighted by the
same closing lines, the real NFL has:

| Statistic | Real NFL | Random leagues | Percentile | p (two-sided) |
|---|---|---|---|---|
| Winning-streak count | 1,338 | 1,408 | 0.0% | < 0.001 |
| Losing-streak count | 1,346 | 1,410 | 0.1% | 0.001 |
| Raw lag-1 autocorrelation | 0.059 | 0.016 | 99.7% | 0.007 |

Fewer streak *segments* across the same number of games means each run is longer.
By these raw measures momentum is unmistakable — teams cluster wins and losses
more than chance.

## Step 2: Why "raw" is the wrong lens

The betting line is not fixed. **It moves after every game**: a win pushes a
team's next-game probability up, a loss pushes it down. So a team's win/loss
string is automatically correlated with its own drifting line, even if nothing
"extra" is going on.

Write each outcome as market probability plus surprise, $Y_t = p_t + r_t$. The
raw game-to-game covariance expands into four pieces:

$$
\text{Cov}(Y_t, Y_{t+1}) = \underbrace{\text{Cov}(p_t, p_{t+1})}_{\text{line drifts smoothly}}
+ \underbrace{\text{Cov}(r_t, p_{t+1})}_{\text{line reacts to results}}
+ \text{Cov}(p_t, r_{t+1}) + \underbrace{\text{Cov}(r_t, r_{t+1})}_{\text{true momentum}}
$$

Only the last term — do *surprises* predict future *surprises* — is momentum the
market missed. The middle term, $\text{Cov}(r_t, p_{t+1})$, is just the market
raising the line after a win. The raw test adds these together and misreads the
sum as momentum.

## Step 3: Remove the line, and it disappears

Subtract each game's probability before measuring correlation. Instead of "did
the team win?" ask "did the team win **more than the line expected**?" — the
residual $r_i = Y_i - p_i$ — and test whether that predicts the next game.

| Lag | Raw autocorr | Raw pct | Residual autocorr | Residual pct | Residual p |
|---|---|---|---|---|---|
| 1 | 0.059 | 99.7% | 0.012 | 80.0% | 0.401 |
| 2 | 0.010 | 23.1% | −0.028 | 3.8% | 0.077 |
| 3 | 0.037 | 87.5% | −0.010 | 25.7% | 0.515 |
| 4 | 0.064 | 99.7% | 0.022 | 90.9% | 0.182 |
| 5 | 0.044 | 95.8% | 0.008 | 67.1% | 0.658 |

The raw column screams momentum. The residual column shows nothing — no lag is
significant once the market line is removed. The apparent persistence *was* the
market pricing each streak in real time.

`output/analysis/team_autocorrelation.png` shows this as two histograms: raw
(real NFL at the far right, 99.7th percentile) and residual (real NFL back in the
middle of the random cloud, 80th percentile).

## Step 4: The predictive confirmation

A correlation test is descriptive; the decisive check is prediction. The
walk-forward test (`scripts/10_market_vs_history.py`) gives a model the market
probability plus recent history and asks whether later games become easier to
predict. They do not — adding history slightly *worsens* out-of-sample log loss,
exactly like the random leagues. No residual correlation to find, no prediction
to gain.

---

## How to read percentile vs p-value

We measure a statistic on the real NFL, then on each of 10,000 simulated leagues.
Those 10,000 values are the "what random looks like" distribution.

- **Percentile** = where the real value sits in that pile. 99.7% means only 0.3%
  of random leagues scored higher; 80% means the real value is unremarkable.
- **p-value (two-sided)** = the probability, if the coin-flip model were true, of
  landing at least this far from the middle in either direction. 99.7th
  percentile → 0.3% above → doubling → p ≈ 0.006–0.007.

They are the same fact in two forms: position, and how surprising that position is.

## A note on the analysis history

Two earlier versions of this test were wrong, in instructive ways:

1. **Pooling bug.** The first version compared the real league-average against a
   pool of *individual* team-seasons from all simulations. An average of 36 teams
   has ~6× less spread than a single team, so the pooled ruler was far too wide
   and the real value looked ordinary (68th percentile). Fixed by computing one
   league-average per simulated league.

2. **Raw-vs-residual mis-specification.** The corrected pooling still measured
   *raw* win/loss autocorrelation, which — as shown above — conflates real
   momentum with the market's own line movement. The right statistic subtracts
   each game's probability first.

Both corrections point to the same conclusion: momentum is real, and the market
has already priced it in.

---

*Reproduce all numbers:* `python scripts/06_analysis_team_timeseries.py`
