# Are NFL Games Just Weighted Coin Flips?

*Testing whether professional football outcomes are statistically distinguishable from independent random draws when conditioned on pregame betting market probabilities.*

---

A bettor isn't really competing against chance. A bettor is competing against a **price** — the odds, which already reflect what the market knows about each game. So the useful question isn't "can you predict the game?" It's a harder one: **can any piece of information help you once it's already baked into the price?**

We test that with NFL betting data. For each of 2,946 games (2011–2021) the closing line implies a home-win probability. We build 10,000 synthetic NFL histories that keep those exact probabilities but decide each game with a weighted coin flip, and ask what, if anything, separates the real NFL from the coin-flip version.

Three findings, in plain terms:

- **NFL games are weighted coin flips, not 50/50 coin flips.** The odds are well-calibrated, and "upsets" happen about as often as the prices say they should.
- **Real results are slightly streakier than the simplest weighted-coin model** — the NFL has about 5% fewer separate winning and losing runs than the random histories, so runs last a little longer. It's a small, real amount of extra clustering, not a dramatic "hot hand."
- **Measured against the price, that extra clustering carries no usable information.** Once you score each result against the same game's market probability, the persistence is gone, and knowing a team's recent form does not improve predictions of future games beyond what the odds already say.

The NFL is the evidence here, so these specific numbers are about the NFL. But the lesson is general to sports betting: being right about a game is not the same as being more right than the price.

$$
Y_i^{\text{NFL}} \overset{?}{\sim} Y_i^* \sim \text{Bernoulli}(p_i)
$$

where $p_i$ is the vig-adjusted closing moneyline probability for game $i$.

---

## The Experiment

For each of 2,946 NFL games (2011–2021 seasons, excluding Super Bowls), we observe:
- The **pregame market probability** $p_i$ from the closing moneyline (vig-adjusted)
- The **actual outcome** $Y_i \in \\{0, 1\\}$

We generate synthetic histories by **keeping** $p_i$ but **replacing** $Y_i$ with $Y_i^* \sim \text{Bernoulli}(p_i)$:

![Experiment Design](output/diagrams/experiment_design.png)

*Figure 1: The experimental design. Each synthetic game has the same teams, date, and pregame probability $p_i$ as the real game, but the outcome is determined by an independent Bernoulli draw rather than the actual result. We simulate 10,000 complete NFL histories this way.*

Each synthetic NFL experiences the **same changing probability environment** as the real league — favorites, underdogs, home field advantage, all reflected in $p_i$ — but outcomes are purely random conditional on those probabilities.

The null hypothesis is **conditional independence**:

$$
H_0: \quad Y_i \perp Y_j \mid \\{p_k\\}_{k=1}^n \qquad \forall i \neq j
$$

If this holds, then once you know the market probability $p_i$ for game $i$, past outcomes $Y_1, \ldots, Y_{i-1}$ contain no additional predictive information. Streaks, momentum, and team-specific patterns should vanish.

---

## First Test: Is the Market Calibrated?

Before testing independence, we verify that the market probabilities $p_i$ are well-calibrated. If $p_i$ systematically over- or under-estimates win probability, the Bernoulli null would be wrong for the wrong reason.

**Calibration** requires that for all $p$:

$$
P(Y_i = 1 \mid p_i = p) = p
$$

We bin games by $p_i$ (in 5% increments) and compare the actual home win rate within each bin to its expected value under the Bernoulli model, using the Monte Carlo distribution to establish confidence intervals.

![Calibration Analysis](output/analysis/calibration.png)

*Figure 2: **Left:** Calibration curve. Each bubble represents a probability bin; bubble size indicates sample size. The red diagonal is perfect calibration. Real NFL outcomes (dark red) fall within the 95% confidence interval (shaded blue) from 10,000 Bernoulli simulations for all 19 bins. **Right:** Calibration residuals (actual – expected). No bin deviates significantly; markets are well-calibrated.*

**Result:** 0 of 19 probability bins fall outside the 95% confidence interval. Markets are well-calibrated across the full range of pregame probabilities — no systematic overestimation or underestimation of win likelihood.

The **average vig-adjusted home win probability** was 56.6%, and **actual home win rate** was 55.8%. The difference (–0.8 percentage points) is statistically consistent with sampling variation (Monte Carlo percentile: 18.3%).

---

## What Defines an Upset?

For a game where the favorite has probability $p_{\text{fav}} = \max(p_i, 1 - p_i)$, the **upset probability** is simply:

$$
P(\text{upset}) = 1 - p_{\text{fav}}
$$

A 70% favorite is **expected** to lose 30% of the time. Such outcomes are not anomalous if probabilities are correctly calibrated; they are the natural consequence of the 30% tail of the Bernoulli distribution.

Below, we compare the **actual frequency of upsets** (underdog wins) to the distribution generated by 10,000 Bernoulli NFL histories:

![Upset Distribution](output/analysis/upsets_total.png)

*Figure 3: Distribution of total upset counts from 10,000 Bernoulli simulations (blue histogram). The real NFL had **997 upsets** (red dashed line), which sits comfortably within the Monte Carlo distribution (MC mean: 978.5 ± 24.6, 95% CI: [930, 1027], percentile: 76.7%). The real NFL is not an outlier.*

The real NFL produced 997 upsets out of 2,946 games (33.8%). The Bernoulli model predicts 978.5 ± 24.6 upsets on average. The real NFL sits at the **76.7th percentile** of the Monte Carlo distribution — well within the expected range.

### Do Strong Favorites Get Upset More or Less Often Than Expected?

One might hypothesize that heavy favorites are **overvalued** (upset more than $1 - p_{\text{fav}}$ predicts) or that true underdogs are **undervalued** (win more often than $p$ suggests). The data:

![Upsets by Favorite Strength](output/analysis/upsets_by_bin.png)

*Figure 4: Upset rates by favorite win probability. Real NFL (red/green bars) compared to theoretical expectation (gray) and Bernoulli Monte Carlo mean (blue) with error bars showing 95% confidence intervals. All 8 bins are consistent with the Bernoulli model (0/8 outside CI). No evidence for favorite-longshot bias or systematic mispricing.*

| Favorite Strength | Real Upset Rate | Bernoulli Expected | Within 95% CI? |
|-------------------|-----------------|-------------------|----------------|
| 50–55% | 49.0% | 47.4% | ✓ |
| 55–60% | 45.5% | 42.5% | ✓ |
| 60–65% | 40.2% | 37.5% | ✓ |
| 65–70% | 34.7% | 32.6% | ✓ |
| 70–75% | 23.6% | 27.4% | ✓ |
| 75–80% | 20.1% | 22.7% | ✓ |
| 80–90% | 15.6% | 16.0% | ✓ |
| 90%+ | 10.0% | 8.3% | ✓ |

**All bins** fall within the 95% confidence interval from Monte Carlo. There is no systematic pattern where strong favorites or longshot underdogs deviate from Bernoulli expectations.

---

## Building 10,000 Random NFL Histories

For each Monte Carlo simulation $m = 1, \ldots, 10{,}000$, we generate a complete NFL season history:

$$
Y_i^{*(m)} \sim \text{Bernoulli}(p_i), \qquad i = 1, \ldots, 2{,}946
$$

Each simulation uses the **exact same** $p_i$ values from the real NFL — same teams, same dates, same market probabilities, same chronological sequence. Only the realized winner is replaced by a coin flip weighted by $p_i$.

These synthetic histories let us ask: **what patterns would we expect to see in a purely random NFL?** And critically: **does the real NFL deviate from that?**

The market probability $p_i$ already encodes:
- Team strength differentials
- Home field advantage  
- Injuries, weather, travel, rest
- Coaching matchups
- Historical performance

We are **not** testing whether these factors matter. We are testing whether, **after the market has already priced them in**, any additional structure remains.

---

## Can We Tell Which NFL Is Real?

This is the core question. Given one real NFL and 10,000 synthetic Bernoulli NFLs, can we identify the real one by examining outcome patterns?

### Market Probabilities Vary Across Games

First, confirm that $p_i$ is not constant. If all games were 50-50 coin flips, the Bernoulli null would be trivial. Instead, market probabilities span a wide range, reflecting the changing competitive landscape:

![Win Probability Distribution](output/exploratory/win_prob_distribution.png)

*Figure 5: Distribution of vig-adjusted home win probabilities $p_i$ across 2,946 games. Mean: 56.6% (reflecting home field advantage). The wide spread shows that games are not uniform coin flips — favorites and underdogs exist. The question is whether outcomes conditional on these $p_i$ are Bernoulli.*

The average home team had a 56.6% win probability, but individual games ranged from 8% to 96%. The Bernoulli model respects this variation — each game $i$ has its own weight $p_i$.

### Real games are a little streakier than the model

Do wins and losses cluster more than weighted coin flips would produce? A little. Across all teams, the real NFL has about **5% fewer separate winning and losing runs** than the average random history — 1,338 winning runs versus a simulated 1,408, and 1,346 losing runs versus 1,410. With the same number of games but fewer runs, each run lasts slightly longer on average.

That is a small, real amount of extra clustering — enough to stand out clearly against 10,000 random histories ($p < 0.001$), but far from a dramatic "hot hand." The single longest streaks tell the same modest story:

![Longest-streak distributions](output/analysis/team_max_streaks.png)

*Figure 6: The longest winning (left) and losing (right) run produced by each random history, with the real NFL marked. The real NFL's longest runs — 21 wins, 20 losses — are toward the high end of what random weighted coin flips already produce, but well inside the range (92nd and 89th percentiles). Takeaway: real streaks are a bit longer than typical, not off the charts.*

We should resist calling this "momentum." All we can safely say is that real results carry a little more persistence than the simplest weighted-coin model. The more important question is whether that persistence is worth anything to a bettor.

### Measured against the price, the persistence disappears

A streak is only useful to a bettor if it tells you something the **odds don't already reflect**. So instead of asking "did the team win?", we score each game against its own price: did the team win *more often than that game's odds implied*? Then we check whether beating the price in recent games says anything about beating the price in the next one.

![Streaks versus the market](output/analysis/team_autocorrelation.png)

*Figure 7: A team's game-to-game consistency, measured two ways, against 10,000 random histories. **Left — raw wins and losses:** the real NFL (red) sits at the far right, reflecting the extra clustering from the previous section. **Right — performance relative to each game's price:** the real NFL falls back into the middle of the random pack. Once you account for what the odds already expected, the apparent pattern is gone.*

| Lag | Raw wins/losses (percentile) | Relative to the price (percentile) |
|----|----|----|
| 1 game apart | 99.7% | 80.0% |
| 2 games apart | 23.1% | 3.8% |
| 3 games apart | 87.5% | 25.7% |
| 4 games apart | 99.7% | 90.9% |
| 5 games apart | 95.8% | 67.1% |

**What this shows:** the raw clustering is real (left column), but it is not a signal the market has missed. Once each result is judged against the price it faced, the real NFL looks like the random histories at every gap. The extra streakiness lives *inside* what the odds already say, not on top of it.

### And it doesn't help predict the next game

The comparison above is descriptive. The decisive test is predictive: start from the market's probability, add a team's recent history, and see whether the *next* game becomes easier to predict — on games the model has never seen. If recent form contained anything the next game's price was missing, this is where it would show up.

We trained only on earlier games and predicted later ones (never the reverse), comparing five models: the market price alone, and the price plus last game, last 3 games, last 5 games, or all recent history together.

![Market vs History](output/analysis/market_vs_history.png)

*Figure 8: How much each extra ingredient changes prediction accuracy versus using the price alone (lower is better; brown bars are the real NFL, gray bands are the range from pure chance). Every version of "add recent history" lands in the chance range — and slightly worse than the price alone, because the extra detail mostly adds noise.*

**What this shows:** knowing a team's recent form does not improve on the price. This is the practical companion to the previous section — there is no leftover signal to find, and none to bet on.

The wins and losses matter. They're just not telling us anything the price doesn't already say.

---

## Summary of Statistical Tests

Each null distribution is built by computing the identical statistic within each of the 10,000 simulated leagues (team-averaged quantities are averaged across teams separately within each league; maxima use the distribution of per-league maxima).

| Test | Real NFL | MC mean | 95% CI | Percentile | Two-sided $p$ | Beats the model? |
|------|----------|---------|--------|------------|---------------|------------|
| **Total home wins** | 1,644 | 1,666 | [1,617, 1,716] | 18.3% | — | No |
| **Calibration bins** | 0 / 19 outside CI | ~1 / 19 | — | — | — | No |
| **Total upsets** | 997 | 978.5 | [930, 1,027] | 76.7% | — | No |
| **Upset bins** | 0 / 8 outside CI | ~0.4 / 8 | — | — | — | No |
| **Win streak count** *(raw)* | 1,338 | 1,408 | [1,369, 1,446] | 0.0% | <0.001 | streaks real |
| **Loss streak count** *(raw)* | 1,346 | 1,410 | [1,372, 1,449] | 0.1% | 0.001 | streaks real |
| **Autocorr lag 1** *(raw)* | 0.059 | 0.016 | [−0.014, 0.047] | 99.7% | 0.007 | streaks real |
| **Autocorr lag 1** *(residual)* | 0.012 | 0.000 | [−0.029, 0.030] | 80.0% | 0.401 | No |
| **Autocorr lag 4** *(residual)* | 0.022 | 0.000 | [−0.032, 0.033] | 90.9% | 0.182 | No |
| **Recent history (out-of-sample)** | worse | ≈ chance | — | — | — | No |

**Verdict:** Raw win/loss sequences are modestly streakier than weighted coin flips (top rows) — real, but small. Once each result is scored against its own game's price, that extra clustering is no longer visible (residual rows), and recent history provides no out-of-sample edge. Whatever the streaks contain, the prices already reflect it.

---

## What Did We Actually Learn?

Return to the central question:

$$
Y_i \mid p_i \overset{?}{\sim} \text{Bernoulli}(p_i)
$$

**What is genuinely there:**
- A small amount of extra clustering. Real NFL runs of wins and losses are about 5% fewer — and therefore a bit longer — than the weighted-coin model produces. It is a modest effect, not a dramatic hot hand, and we can't attribute it to any specific cause.

**What the market already reflects:**
- ✓ Well-calibrated probabilities (0/19 calibration bins outside CI)
- ✓ Upsets at the predicted rate (76.7th percentile; 0/8 bins outside CI)
- ✓ Scored against each game's own price, the extra clustering is no longer detectable (residual comparison not significant at any lag)
- ✓ Recent history gives no out-of-sample predictive edge (walk-forward test)

**Why both can be true.** A team's raw record and the prices it is given both reflect the same underlying strength, so raw win/loss sequences naturally look clustered. But when we measure each result against the price it actually faced, the extra clustering is already contained in those prices — there is nothing left over to predict the next result.

**What this does NOT mean:**
- ❌ It does **not** mean games are "random" — skill, coaching, and injuries clearly shape outcomes. The prices reflect them.
- ❌ It does **not** mean the market is perfect — only that no simple public signal (streaks, recent form) improved on it in this NFL sample.
- ❌ It does **not** prove "momentum" in a causal sense; we can only say real results show a little more persistence than the simplest model.

**What this DOES mean:**
- After accounting for the closing price, past outcomes carry **no measurable additional predictive information** in this data.
- Being right about a game and beating the price are different things. A real pattern that is already in the odds is not an edge.

**A note on the analysis history.** Two earlier iterations of the streak test were wrong in instructive ways. The first compared the real league against a mismatched random baseline, which understated the natural variation and made the effect look larger. The second measured raw wins and losses, which conflates a team's clustering with what its game prices already reflected. The comparison used here scores each result against its own game's price and is backed up by the out-of-sample prediction test. See `docs/persistence_explained.md`.

---

## The Roulette Analogy

A roulette wheel has a **known probability distribution** but **unpredictable individual outcomes**. Even if you know that red has exactly 18/38 probability, you cannot predict the next spin better than that.

The market hypothesis is analogous, with one key difference: **the wheel's weighting changes for every game**.

$$
p_1, p_2, p_3, \ldots, p_n
$$

Each $p_i$ reflects:
- Team strength (which itself evolves)
- Home field advantage
- Injuries, weather, rest, travel
- Matchup-specific factors
- Historical trends

The market may be **excellent at estimating these changing weights** — better than any individual analyst — without being able to **predict the individual realization** beyond the probability itself.

Our NFL results fit this picture. Each game's price already carries the information you might hope to use — team strength, recent form, matchup — so once you know the current weighting (the line), the next spin looks unpredictable. The prices track the changing odds well enough that the leftover, game-to-game, behaves like chance.

---

## Implications for Sports Betting

The *principle* here isn't specific to football, even though our evidence is. Whenever a market's price is close to the true probability and no extra information improves on it, the odds offered to a bettor are set slightly worse than that true probability. This margin — the "vig," or the sportsbook's cut — applies to every wager regardless of result. The NFL data below is a concrete illustration of what that margin does; it is not a claim that every sport's market behaves identically.

A common assumption is that a bettor who wins the majority of wagers must be profitable. The margin makes this false, and the mechanism is clearest in dollar terms.

### An even matchup (–110 both sides)

For two evenly matched teams, a sportsbook typically requires a risk of **\$110 to win \$100** on either side. Across 100 such bets at an exact 50% win rate:

| Outcome | Result |
|---|---|
| 50 wins | +\$5,000 |
| 50 losses | –\$5,500 |
| **Net** | **–\$500** |

A perfect coin-flip record still produces a \$500 loss. Break-even at these odds requires a **52.4%** win rate, not 50%.

### A heavy favorite (an 80/20 matchup)

A team with a true 80% win probability corresponds to fair odds of –400 (risk \$400 to win \$100). A sportsbook shades this to approximately –450, so \$100 at risk returns only **\$22.22** in profit. Across 100 such bets at an exact 80% win rate:

| Outcome | Result |
|---|---|
| 80 wins | +\$1,778 |
| 20 losses | –\$2,000 |
| **Net** | **–\$222** |

An 80-of-100 record still produces a \$222 loss. Because a single loss offsets 4.5 wins, break-even rises from 80% to **81.8%** — above the true win probability.

### Strategies applied to the full dataset

To confirm the effect is not an artifact of two chosen examples, four flat-stake strategies were applied to all 2,946 games (2011–2021) at \$100 per bet, and to 10,000 matched Bernoulli simulations for comparison.

![Profit distribution relative to break-even](output/analysis/betting_distribution.png)

*Figure: Total profit from betting every favorite at \$100 flat over eleven seasons. The blue histogram is 10,000 simulated leagues in which outcomes are weighted coin flips. Almost none finish above \$0 (green line). The observed NFL result (red line) falls within the losing distribution.*

| Strategy | Bets | Observed result | Return | Simulation mean |
|---|---:|---:|---:|---:|
| Bet every favorite | 2,946 | **–\$14,509** | –4.9% | –3.7% |
| Bet even matchups (45–55%) | 367 | **–\$2,650** | –7.2% | –4.5% |
| Bet heavy favorites (≥75%) | 655 | **–\$1,268** | –1.9% | –3.7% |
| Bet every underdog | 2,946 | **–\$10,231** | –3.5% | –3.7% |

![Cumulative profit by strategy](output/analysis/betting_bankroll.png)

*Figure: Cumulative profit over eleven seasons for four flat-stake strategies. Every trajectory trends downward. No selection rule tested — favorites, underdogs, even matchups, or heavy favorites — returns to break-even.*

The simulation mean is approximately **–3.7% for every strategy**, matching the average sportsbook margin in this data (3.8%). The expected return of any strategy equals the negative of the vig. Favorites, underdogs, and even matchups converge to the same losing rate because, beyond the price, the outcome carries no exploitable information and the margin is deducted from every wager. The heavy-favorite strategy returned above its simulation mean yet still lost \$1,268, illustrating that a high win rate does not imply a profit.

### Why losses are guaranteed over enough bets

Over a finite number of bets, variance allows some outcomes above break-even; a fraction of the 10,000 simulated leagues finish in profit. This mirrors a roulette wheel: any single session may win, but each spin carries negative expected value. As the number of bets grows, the law of large numbers drives the realized return toward its expectation. Because that expectation is negative for every strategy, sustained betting converges to a loss. No selection rule reverses this, since the profitable outcomes are produced by chance rather than by any repeatable edge.

---

## Methods & Data

### Dataset
- **Source:** [Sportsbook Review Historical Data](https://github.com/flancast90/sportsbookreview-scraper) (2011–2021)
- **Sample:** 2,946 NFL games (regular season + playoffs)
- **Exclusions:** 10 Super Bowls (neutral site, ambiguous "home" team)
- **Odds type:** Closing moneyline (pregame, not live)
- **Vig adjustment:** Normalized probabilities to sum to 1.0

### Probability Calculation
American moneyline odds $m$ are converted to implied probability:

$$
p_{\text{implied}} = 
\begin{cases}
\frac{100}{m + 100} & \text{if } m > 0 \\\\
\frac{|m|}{|m| + 100} & \text{if } m < 0
\end{cases}
$$

Since $p_{\text{home}} + p_{\text{away}} > 1$ (the "vig"), we normalize:

$$
p_{\text{home, vig-free}} = \frac{p_{\text{home}}}{p_{\text{home}} + p_{\text{away}}}
$$

Average vig: **3.83%** (median: 3.79%, range: 1.3%–41.2%).

### Monte Carlo Simulation
- **Simulations:** 10,000 complete NFL histories
- **Method:** For each simulation $m$ and game $i$, draw $Y_i^{*(m)} \sim \text{Bernoulli}(p_i)$
- **Seed:** 42 (reproducible)
- **Compute time:** ~3 minutes on standard CPU

### Sample Size by Season

![Games by Season](output/exploratory/games_by_season.png)

*Figure 9: Number of games per season. Consistent coverage (267–284 games/season) across 11 NFL seasons.*

### Code & Reproducibility
- **Language:** Python 3.12
- **Dependencies:** NumPy, Matplotlib, SciPy
- **Repository:** [github.com/chrissmithphd/just-a-coin-flip](https://github.com/chrissmithphd/just-a-coin-flip)
- **License:** MIT

Run the full analysis pipeline:
```bash
git clone https://github.com/chrissmithphd/just-a-coin-flip.git
cd just-a-coin-flip
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/08_run_all_analyses.py
```

---

## Additional Results

### Autocorrelation by Lag: Raw vs. Adjusted

![Autocorrelation by lag](output/analysis/team_autocorr_by_lag.png)

*Figure 10: Team consistency at lags 1–5, measured two ways. Raw win/loss (red) rides above the random range at several lags; measured against each game's price (green), every lag falls inside the random band. The apparent pattern is already contained in the prices at every lag, not just one game apart.*

### Game-Level Residual Autocorrelation

![Residual Autocorrelation](output/analysis/residuals_autocorrelation.png)

*Figure 11: Autocorrelation of residuals $r_i = Y_i - p_i$ on the global game sequence, lags 1–10. All lags fall within the ±2σ range — no serial structure once the line is removed. (This mixes all teams, so it complements the team-level residual test in Test 2.)*

### Residual Distribution

![Residual Distribution](output/analysis/residuals_distribution.png)

*Figure 12: Distribution of residuals $r_i = Y_i - p_i$. Real NFL (red) overlays the Bernoulli Monte Carlo distribution (blue). Distributions are nearly identical (real: mean = –0.008, std = 0.474; MC: mean = 0.000, std = 0.474).*

---

## Future Work

- [ ] **Extend to 2022–2024:** Add recent seasons (post-legalization era, COVID recovery)
- [ ] **Cross-sport comparison:** Test NBA, MLB, NHL with identical framework
- [ ] **Conditional analyses:**
  - Division vs. non-division games
  - Playoff games (higher stakes)
  - Prime-time vs. early games
  - Outdoor games with extreme weather
- [ ] **Within-team tests:** Per-team hypothesis tests (32 independent tests with Bonferroni correction)
- [ ] **Bayesian calibration:** Hierarchical model for $P(Y \mid p)$
- [ ] **Favorite-longshot bias:** Deep dive on extreme underdogs ($p < 0.15$)
- [ ] **Live betting lines:** Test whether in-game updates remain calibrated

---

## References

- **Efficient markets hypothesis in sports:** Thaler, R. H., & Ziemba, W. T. (1988). Anomalies: Parimutuel Betting Markets. *Journal of Economic Perspectives*, 2(2), 161–174.
- **Market-based probabilities:** Wolfers, J., & Zitzewitz, E. (2004). Prediction Markets. *Journal of Economic Perspectives*, 18(2), 107–126.
- **Data source:** [Sportsbook Review scraper](https://github.com/flancast90/sportsbookreview-scraper) by flancast90

---

## Contact

**Chris Smith** — [@chrissmithphd](https://github.com/chrissmithphd)  
**Project:** [github.com/chrissmithphd/just-a-coin-flip](https://github.com/chrissmithphd/just-a-coin-flip)

---

*Generated with [Claude Code](https://claude.com/claude-code)*
