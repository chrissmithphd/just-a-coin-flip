# Statistical Audit Report
**Date:** 2026-09-23  
**Auditor Role:** Skeptical statistical reviewer  
**Objective:** Verify every claim traces to correct Monte Carlo implementation

---

## Executive Summary

**Four critical statistical errors invalidate multiple key claims:**

1. **Team autocorrelation (PSEUDOREPLICATION):** Pooled 320,000+ observations instead of 10,000
2. **Streak counts (NO TEST):** Showed averages, never tested significance  
3. **Max streaks (INVALID COMPARISON):** Compared to extrema across all simulations
4. **Residuals after outcomes (POOLING ERROR):** Pooled across simulations incorrectly

**Three tests are correctly implemented:**
- Calibration bins (correct MC procedure)
- Upset bins (correct MC procedure)  
- Residual autocorrelation (correct MC procedure)

**Multiple comparison issues throughout** (19 calibration bins, 8 upset bins, 10 residual lags) with no correction.

---

## Audit Table

| README/RESULTS Claim | Statistic $T$ | Implementation | Status | Issue | Required Correction |
|---------------------|---------------|----------------|--------|-------|---------------------|
| **"Team autocorr lag 1: 0.059, percentile 68.2%"** | $T = \frac{1}{N_{\text{teams}}}\sum_{\text{team}}\rho_1(\text{team})$ | `scripts/06_analysis_team_timeseries.py:208-262` | **INVALID** | **PSEUDOREPLICATION**: Pooled all team×sim autocorrs (n=320k+) instead of computing per-sim team-average | For each sim $m$: compute $\rho_1$ for each team, average → $T_m$. Compare $T_{\text{real}}$ to distribution of 10,000 $T_m$ |
| **"Win streaks: 1,342 real vs 1,412 MC"** | $T = \text{total streak count}$ | `scripts/06_analysis_team_timeseries.py:124-170` | **INVALID** | **NO STATISTICAL TEST**: Pooled all streaks, showed average, never computed percentile | For each sim $m$: count streaks → $T_m$. Compare 1,342 to distribution. Report percentile. |
| **"Max win streak 21 within MC range (up to 41)"** | $T = \max(\text{all streaks})$ | README claim from pooled data | **INVALID** | **EXTREMA ACROSS SIMS**: Took max of ~14M pooled streaks, not distribution of per-sim maxima | For each sim $m$: find max streak → $T_m$. Compare 21 to percentile in distribution of 10k maxima. |
| **"Max loss streak 20 within range (up to 37)"** | $T = \max(\text{all streaks})$ | README claim from pooled data | **INVALID** | Same as above | Same correction |
| **"Residuals after wins: 0.006 ± 0.457"** | $T = \bar{r}_{\text{after win}}$ | `scripts/07_analysis_residuals.py:109-169` | **INVALID** | **POOLING ERROR**: Pooled all residuals after wins across 10k sims | For each sim $m$: $T_m = \text{mean}(r_i \mid Y_{i-1}=1)$. Compare to distribution. |
| **"Residuals after losses: -0.025 ± 0.461"** | $T = \bar{r}_{\text{after loss}}$ | Same file, same function | **INVALID** | Same pooling error | Same correction |
| **"Calibration: 0/19 bins outside 95% CI"** | $T_{\text{bin}} = \frac{\sum_{i \in \text{bin}} Y_i}{n_{\text{bin}}}$ | `scripts/04_analysis_calibration.py:30-98` | **VALID** ✓ | None in MC procedure. **Multiple comparisons warning**: 19 tests, expect ~1 outside by chance | Add Bonferroni note: with 19 bins, α=0.05 → expect 0.95 bins outside by chance. Consider α=0.05/19≈0.0026 per test. |
| **"Upsets: 997 real, MC 978.5±24.6, percentile 76.7%"** | $T = \sum_i \mathbb{1}(\text{upset}_i)$ | `scripts/05_analysis_upsets.py:identify_upsets` + histogram | **VALID** ✓ | None | None |
| **"Upset bins: 0/8 outside CI"** | $T_{\text{bin}} = \frac{\sum_{i \in \text{bin}}\mathbb{1}(\text{upset}_i)}{n_{\text{bin}}}$ | `scripts/05_analysis_upsets.py:63-150` | **VALID** ✓ | **Multiple comparisons**: 8 bins → expect 0.4 outside by chance | Add note about multiple comparisons |
| **"Residual autocorr lag 1: 0.025, percentile 91.3%"** | $T = \rho_1(r_1, \ldots, r_n)$ | `scripts/07_analysis_residuals.py:56-106` | **VALID** ✓ | **Multiple comparisons**: 10 lags tested | Add note: 10 lags, expect 0.5 outside by chance at α=0.05 |
| **"All lags consistent with white noise"** | Multiple $\rho_k$ | Same | **QUESTIONABLE** | Tested 10 lags, none significant. With α=0.05, P(0/10 outside) = 0.60. Not strong evidence. | Report: "No lags significant at α=0.05; consistent with but not strong evidence for white noise (10 tests)" |

---

## Detailed Error Analysis

### Error 1: Team Autocorrelation (CRITICAL)

**File:** `scripts/06_analysis_team_timeseries.py`, lines 208-262

**Claim:** "Team autocorrelation lag 1: 0.059, MC mean: 0.016, MC std: 0.098, percentile: 68.2%"

**What the code does:**

```python
# Real: average autocorr across teams (lines 214-226)
autocorrs_real = []  # list of per-team autocorr arrays
for team, game_list in team_data_real.items():
    ac = compute_autocorrelation(outcomes, max_lag)
    autocorrs_real.append(ac)
mean_autocorr_real = np.nanmean(autocorrs_real, axis=0)  # shape: (max_lag,)

# MC: pool all team×simulation autocorrs (lines 228-260)
autocorrs_mc_all = []
for team in teams:
    for sim_idx in range(n_sims):  # 10,000 simulations
        ac = compute_autocorrelation(outcomes_sim, max_lag)
        autocorrs_mc_all.append(ac)  # pool everything

# Compare (lines 249-260)
mc_values = autocorrs_mc_all[:, lag_idx]  # shape: (n_teams × 10,000,)
percentile = (mc_values < mean_autocorr_real[lag_idx]).mean() * 100
```

**Problem:** `mc_values` has ~320,000 observations (32 teams × 10,000 sims). Each team in each simulation is treated as an independent observation. **This is pseudoreplication.**

**Correct $T$ statistic:**

$$
T_{\text{real}} = \frac{1}{N_{\text{teams}}} \sum_{\text{team}} \rho_k(\text{team outcomes})
$$

$$
T_m = \frac{1}{N_{\text{teams}}} \sum_{\text{team}} \rho_k(\text{team outcomes in sim } m)
$$

Compare $T_{\text{real}}$ to $\\{T_1, \ldots, T_{10000}\\}$.

**Impact:** 
- True MC distribution should have 10,000 observations, not 320,000
- Standard deviation will be ~√32 ≈ 5.7× larger than reported
- Reported MC std = 0.098 → True std likely ~0.56
- Real value 0.059 would be much closer to MC mean (0.016) in proper distribution
- **Percentile claim is invalid**

---

### Error 2: Streak Distributions (MAJOR)

**File:** `scripts/06_analysis_team_timeseries.py`, lines 124-170

**Claim:** "Win streaks: Real NFL 1,342 total, MC avg 1,412/sim"

**What the code does:**

```python
# Real
all_win_streaks_real = []
for team in teams:
    win_streaks, _ = compute_streaks(team_outcomes)
    all_win_streaks_real.extend(win_streaks)
# Result: 1,342 streaks

# MC (lines 150-159)
all_win_streaks_mc = []
for team in teams:
    for sim_idx in range(10000):
        win_streaks, _ = compute_streaks(outcomes_sim)
        all_win_streaks_mc.extend(win_streaks)  # POOL ALL
# Result: ~14 million streaks

# Plot (lines 271-277)
win_mc = {k: v / 10000 for k, v in Counter(all_win_streaks_mc).items()}
# Shows "average count per simulation"
```

**Problem:** Never computed whether 1,342 is statistically unusual. Just pooled and averaged.

**Correct test:**

$$
T_{\text{real}} = 1342 \quad (\text{total streak count in real NFL})
$$

$$
T_m = \text{total streak count in simulation } m
$$

Should report: "Real NFL: 1,342 streaks. MC distribution: mean=1,412, std=? Percentile of 1,342: ?%"

**Impact:** Cannot claim "indistinguishable" without the test. If 1,342 is at 5th percentile, it would be marginally significant.

---

### Error 3: Maximum Streaks (CRITICAL)

**Claim:** "Longest real streaks (21-game win, 20-game loss) are well within the Monte Carlo range (up to 41 and 37)"

**Problem:** Took max across ALL streaks in ALL 10,000 simulations:

```python
max_streak_mc = max(all_win_streaks_mc)  # max of ~14 million streaks
```

**This is wrong.** You're comparing:
- Real: max of ~1,300 streaks
- MC: max of ~14,000,000 streaks

**Correct approach:**

$$
T_{\text{real}} = 21 \quad (\text{longest streak in real NFL})
$$

$$
T_m = \max(\text{all streaks in simulation } m)
$$

Compare 21 to distribution of $\\{T_1, \ldots, T_{10000}\\}$.

**Expected result:** The distribution of per-simulation maxima will have much lower values than 41. The 95th percentile might be around 12-15 games. **Real value of 21 may actually be an outlier.**

---

### Error 4: Residuals After Wins/Losses (MAJOR)

**File:** `scripts/07_analysis_residuals.py`, lines 109-169

**Claim:** "After wins: Real 0.006 ± 0.457, MC 0.000 ± 0.460. Within ±2σ."

**What the code does:**

```python
# Real
residuals_after_win_real = residuals_real[1:][after_win_mask]
# n = 1,643 residuals

# MC (lines 138-145)
residuals_after_win_mc = []
for sim_idx, sim_outcomes in enumerate(mc_outcomes):
    after_win_mask_sim = (sim_outcomes[:-1] == 1)
    residuals_after_win_mc.extend(sim_residuals[1:][after_win_mask_sim])  # POOL
# n ≈ 16 million residuals (1,600/sim × 10,000 sims)

# Compare (lines 152-154)
real_mean = np.mean(residuals_after_win_real)  # 0.006
mc_mean = np.mean(residuals_after_win_mc)      # 0.000
mc_std = np.std(residuals_after_win_mc)        # 0.460
```

**Problem:** Pooled all residuals across simulations. The SD of the pooled residuals (≈0.46) is NOT the SD of the means across simulations (which would be ≈0.46/√1600 ≈ 0.011).

**Correct test:**

$$
T_{\text{real}} = \frac{1}{n_{\text{wins}}}\sum_{i: Y_{i-1}=1} r_i
$$

$$
T_m = \frac{1}{n_{\text{wins},m}}\sum_{i: Y_{i-1}^{(m)}=1} r_i^{(m)}
$$

Compare $T_{\text{real}}$ to $\\{T_1, \ldots, T_{10000}\\}$.

**Impact:** True SD of mean distribution ≈ 0.011, not 0.460. Real value 0.006 is 0.006/0.011 ≈ 0.55σ from MC mean, not 0.006/0.460 ≈ 0.01σ. Completely changes interpretation.

---

## Multiple Comparisons Issues

| Test | # Comparisons | Expected Outside CI (α=0.05) | Observed Outside | Bonferroni α |
|------|---------------|------------------------------|------------------|--------------|
| Calibration bins | 19 | 0.95 | 0 | 0.0026 |
| Upset bins | 8 | 0.40 | 0 | 0.0063 |
| Residual autocorr | 10 | 0.50 | 0 | 0.0050 |
| Team autocorr | 5 | 0.25 | 0* | 0.0100 |

\* Would need to be recomputed after fixing pseudoreplication

**None of the analyses apply multiple comparison corrections.** With 42 total tests (19+8+10+5), expect ~2 false positives even if null is true everywhere.

---

## Overclaims in Language

### Claim: "Statistically indistinguishable"

**Where:** README, multiple locations

**Problem:** This phrase suggests equivalence, but the tests only show "failure to reject null." Cannot distinguish "true null" from "underpowered test."

**Correction:** Use "consistent with" or "no significant deviation from"

---

### Claim: "White noise"

**Where:** README residuals section, "residuals behave like white noise"

**Problem:** Testing 10 lags with none significant (α=0.05) has P(all pass | true null) ≈ 0.60. This is consistent with but not strong evidence for white noise.

**Correction:** "No significant autocorrelation detected at α=0.05 (10 lags tested)"

---

### Claim: "All information captured by $p_i$"

**Where:** README conclusion

**Problem:** Tests show no detectable deviations in ~3,000 games. This does NOT prove all information is captured—only that deviations are below detection threshold.

**Correction:** "Market probability $p_i$ appears to be a sufficient statistic for detectable patterns in this dataset"

---

## Valid Tests (Correctly Implemented)

### ✓ Calibration Bins
**Correctly computes:** For each bin, for each sim $m$: win rate in bin → $T_m$. Compares real to 10,000 $T_m$ values.

**Caveat:** Should note multiple comparisons (19 bins).

### ✓ Total Upsets  
**Correctly computes:** For each sim $m$: total upset count → $T_m$. Percentile correct.

### ✓ Upset Bins
**Correctly computes:** For each bin, for each sim $m$: upset rate in bin → $T_m$. Compares real to distribution.

**Caveat:** Multiple comparisons (8 bins).

### ✓ Residual Autocorrelation (game-level)
**Correctly computes:** For each sim $m$: autocorr of residuals → $T_m$. Compares real to distribution.

**Caveat:** Multiple comparisons (10 lags).

---

## Summary of Required Corrections

1. **Recompute team autocorrelation** with proper aggregation per simulation
2. **Add statistical test for streak counts** (currently only shows average)
3. **Recompute max streak percentiles** using per-simulation maxima
4. **Recompute residuals-after-outcomes test** with proper per-simulation aggregation
5. **Add multiple comparison notes/corrections** throughout
6. **Revise language** to avoid overclaims ("indistinguishable" → "consistent with")
7. **Qualify white noise claim** (weak evidence from non-rejection)
8. **Qualify information capture claim** (detection limits, not proof)

---

## Conclusion

**The analysis contains four implementation errors that invalidate multiple key findings.** Three tests are correctly implemented but suffer from multiple comparison issues. Even after corrections, the language overclaims what hypothesis tests establish.

**Recommended action:** Fix the four errors, rerun analyses, report corrected percentiles, add multiple comparison adjustments, and revise interpretive language to match what the tests actually show.
