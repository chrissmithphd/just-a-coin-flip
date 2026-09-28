# The Team-Persistence Finding, Explained and Verified

This note explains, in plain terms, the one place where the real NFL departs
from the "weighted coin flip" model, how to read the numbers, and what the bug
was that originally hid the effect. Every number here is reproduced by
`scripts/13_verify_persistence.py`.

---

## The finding in one sentence

After a win, an NFL team is a few percentage points more likely to win its next
game than the betting-market probability alone would predict — a small but
statistically real "persistence" that the market does not fully price in.

---

## How to read the two numbers

We measure a statistic on the real NFL, then measure the identical statistic on
each of 10,000 simulated leagues where every game is a coin flip weighted by the
market probability. Those 10,000 values are the "what random looks like"
distribution.

For lag-1 team autocorrelation:

| Quantity | Value |
|---|---|
| Real NFL | 0.059 |
| Average of the 10,000 random leagues | 0.016 |
| Standard deviation of the random leagues | 0.015 |
| Percentile of the real value | 99.7% |
| Two-sided p-value | 0.007 |

**Percentile = 99.7%** means 99.7% of the random leagues scored *lower* than the
real NFL. Only 0.3% scored higher. The real league sits at the extreme top of
the random pile — it is more persistent than almost any coin-flip league.

**p-value = 0.007** restates the same fact as a probability: if the coin-flip
model were true, a value this far from the middle (in either direction) would
occur about 0.7% of the time. Percentile is *where* the real value sits;
p-value is *how surprising* that position is.

They are linked: 99.7th percentile → 0.3% above → doubling for a two-sided test
→ p ≈ 0.006–0.007.

The random average is 0.016, not 0, because strong teams carry a high win
probability across many games; even pure coin flips weighted by those
probabilities produce slight positive correlation. The real signal is the
*excess* over that baseline: 0.059 − 0.016 ≈ **0.04**.

---

## What an autocorrelation of 0.059 actually means

Autocorrelation is a coefficient from −1 to +1, not a probability. But for a
win/loss sequence it has an exact, intuitive translation:

$$
\rho_1 = P(\text{win} \mid \text{won last game}) - P(\text{win} \mid \text{lost last game})
$$

So 0.059 means a team is about **6 percentage points** more likely to win after
a win than after a loss. About 1.6 points of that is the market-driven baseline
(good teams stay good), leaving roughly **4 points of genuine streakiness**
beyond what the price reflects.

This identity is not just theory. `scripts/13_verify_persistence.py` computes
both quantities for every team and confirms they are equal:

```
Packers      autocorr=+0.0062   gap=+0.0062
Saints       autocorr=+0.0201   gap=+0.0201
Ravens       autocorr=-0.0065   gap=-0.0065
Buccaneers   autocorr=+0.2287   gap=+0.2283
```

Running the whole test on the win-rate gap instead of the autocorrelation gives
the same answer (real 0.0585, 99.7th percentile, p = 0.007) — an independent
confirmation through a different statistic.

**Lags are spacings, not additions.** "Lag-4" means a game and the one four
games later; it is the same sequence measured at a wider spacing, not a separate
chance to add on. Lag-4 is also elevated because a hot or cold stretch spans
several games, so games one apart and four apart both look correlated. The
measurements overlap; they describe one tendency, not several to sum.

---

## What the bug was

The test averages a per-team statistic across the ~36 teams. The real NFL gives
one number: average each team's autocorrelation, done.

The **correct** null does the same thing inside each simulated league: for
league *m*, average its teams' autocorrelations into a single value $T_m$. That
gives 10,000 league-level values, and we ask where the real value falls among
them.

The **bug** threw every team from every simulation into one giant bucket —
about 36 × 10,000 ≈ 360,000 individual team values — and compared the real
league-average against that pool.

Why that is wrong: the pool measures the spread of *individual team-seasons*,
which is large. The real number is an *average of 36 teams*, whose spread is
much smaller — smaller by roughly $\sqrt{36} = 6$. Using the pool's wide spread
as the yardstick makes an extreme value look ordinary.

`scripts/13_verify_persistence.py` reproduces this exactly:

| | Std used as yardstick | Real value lands at |
|---|---|---|
| Bugged pooling (360k values) | 0.098 | 68.2nd percentile ("looks normal") |
| Correct per-league null (10k values) | 0.015 | 99.7th percentile ("extreme") |
| Ratio | 6.3× | — |

The 6.3× inflation matches the predicted $\sqrt{36} \approx 6$. The bug widened
the ruler by a factor of six, which is precisely why a real effect looked like
noise.

---

## Why it still doesn't matter for betting

The effect is statistically real but small (~4 points of excess streakiness).
The walk-forward, out-of-sample test (`scripts/10_market_vs_history.py`) shows
that adding recent history does not improve prediction of future games, and the
effect is far too small to overcome the sportsbook margin. It is detectable with
2,946 games and 10,000 simulations; it is not exploitable.

**Statistically significant, practically negligible** — the distinction is the
whole point.

---

*Reproduce all numbers:* `python scripts/13_verify_persistence.py`
