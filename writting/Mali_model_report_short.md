# Identifying Drivers of Internal Displacement in Mali
### A Panel Regression Analysis with Random Forest Validation

---

## Overview

We model monthly changes in log-IDP counts across Mali's 50 admin2 cercles
(2014–2024) as a function of three candidate driver families: **conflict**,
**climate**, and **food prices**. Our goal is to identify which drivers
causally matter, in what form, and at what lag — while being honest about
what a panel regression can and cannot establish.

The analysis proceeds in two parts:

1. **Model selection** (Stages A–E): a structured funnel that narrows 282
   candidate features to a 14-variable final model, growing the specification
   from 1 feature to the final model one step at a time.
2. **Validation and extensions** (Parts II–IV): Random Forest stress-tests
   the linear model, heterogeneity tests reveal what the average effects hide,
   and the independent refugee-outflow outcome cross-validates the findings.

**Data.** IDP source: DTM survey panel, N<5 zero-run-filtered and re-stitched
(`delta_idp_zrt5.csv`). Sample: n = 1,417 observations, 44 cercles, after
requiring complete data for all M5 features. Conflict: ACLED strict (direct
lethal violence). Climate: ERA5 reanalysis. Food: WFP price monitoring.
Spatial weights: road-distance decay matrix (`weighting_beta1.csv`).

**Notation.**

- $y_{it}$ = monthly change in log-IDP (`y_change_per_month`) for cercle $i$ at
  survey date $t$
- $C_{it}$ = controls: `log1p_idp_t1` (AR anchor), `months_between`
- $\alpha_i,\gamma_t$ = cercle and time fixed effects; SE clustered on cercle
- $f(\cdot)$ = feature transform: `log1p` for non-negative counts, identity for
  signed anomalies
- $Wx$ = spatial-spillover twin of $x$ (neighbour-weighted average at the same
  lag band)

**Three criteria for variable inclusion** (applied throughout):
1. **ΔR²_within** — does the variable add explanatory power?
2. **p-value** — is the signal distinguishable from noise?
3. **Coefficient sign** — is the direction mechanistically coherent?

No variable earns its place on one criterion alone.

---

### Feature construction

**Outcome.** The regression target is `y_change_per_month` = (log1p(IDP_t2) −
log1p(IDP_t1)) / months_between — the average monthly log-IDP growth over each
survey interval. Dividing by `months_between` makes observations comparable
regardless of survey gap length (intervals range from 1 to 6 months).

**Transforms.** The transform for each feature is chosen by value distribution:

- **Conflict** (`fatalities`, `events`): always non-negative counts → **log1p**
  transform. This compresses the heavy right tail (rare extreme events) and
  makes the relationship approximately linear on a proportional scale.
- **Climate**: all 112 climate local features use **identity (no transform)**.
  Most are signed anomalies (SPI, temperature, rainfall, streamflow, NDVI,
  soil moisture) that can be negative, so log1p is not applicable. The SPI-derived
  duration and severity variables (`spi6_flood_duration`, `spi6_drought_severity`,
  etc.) are non-negative, but are kept on identity via an explicit override —
  these are already-derived intensity measures where log1p would distort the
  natural scale and reduce |t| empirically.
- **Food**: `inflation_food_price_index` (a growth rate, signed) → identity.
  `c_food_price_index` and `c_maize` (price levels, non-negative) → **log1p**.

Cumulative features (total fatalities over K months) were considered in early
exploratory work but not used — they overlap with adjacent banded windows and
the banded design already captures sustained exposure via multiple
non-overlapping bands.

**Lag structure — banded windows.** Rather than point lags (the value at exactly
N months ago), we use **non-overlapping banded window averages**. A band
`[lo, hi]m` takes the mean of the monthly variable over the window from
`month_of(t1) − hi` to `month_of(t1) − lo`, strictly before the observation
start `t1`. For example:

| Band | window | interpretation |
|---|---|---|
| 1–3m | months 1, 2, 3 before t1 | short-term pre-survey |
| 7–9m | months 7, 8, 9 before t1 | medium-term |
| 10–12m | months 10, 11, 12 before t1 | ~one year lag |
| 13–18m | months 13–18 before t1 | slow-burn effects |
| 19–24m | months 19–24 before t1 | long-term |

Banded windows are **non-overlapping** — the 1–3m and 4–6m bands contain
different months, so they can be included jointly without multicollinearity from
overlap. This is why distributed-lag models (including all bands simultaneously)
remain feasible up to the point where between-band correlations become a problem.

Conflict uses bands 1–3m, 4–6m, 7–9m, 10–12m (no longer lags — conflict is not
expected to have displacement effects at 2+ year horizons). Climate and food use
the same four short bands plus 13–18m and 19–24m (slow-burn pathways exist for
climate). The **current window** (0 months lag, i.e. the observation period
`(t1, t2]` itself) is used only for climate and food — conflict in the current
window is excluded because it risks reverse causality (displacement events may
*cause* conflict rather than vice versa).

**Cumulative windows.** Cumulative features (e.g. total fatalities over the past
12 months) were considered in early exploratory work but **not used in the final
analysis**. They overlap with adjacent banded windows, creating multicollinearity,
and the banded design already captures sustained exposure via multiple
non-overlapping bands. The final feature set contains banded windows only.

**Feature count.** The full candidate set has **282 features**: 16 conflict
(2 indicators × 4 bands × local + spillover), 224 climate (16 variables × 7
lag windows × local + spillover), and 42 food (3 variables × 7 lag windows ×
local + spillover). The indicators are:

| Family | Indicators | Transform | Lag windows |
|---|---|---|---|
| **Conflict** | `fatalities`, `events` (ACLED strict) | log1p | bands 1–3, 4–6, 7–9, 10–12m |
| **Climate** | `spi6`, `spi6_flood_dur`, `spi6_flood_sev`, `spi6_drought_dur`, `spi6_drought_sev`, `p_anom_1m`, `p_anom_3m`, `t2m_anom_1m`, `t2m_max_anom_1m`, `ndvi_anom_1m`, `ndvi_annual_change`, `ndvi_calendar_month_change`, `sm1_anom_1m`, `sm3_anom_1m`, `sm4_anom_1m`, `streamflow_anom_1m` | identity (signed anomalies) | current + bands 1–3, 4–6, 7–9, 10–12, 13–18, 19–24m |
| **Food** | `c_maize`, `c_food_price_index`, `inflation_food_price_index` | identity or log1p (level vs rate) | current + bands 1–3, 4–6, 7–9, 10–12, 13–18, 19–24m |

Each local feature has a **spatial-spillover twin** `W·X` constructed as the
row-standardised, zero-diagonal product of the road-distance weight matrix and
the panel of local values, pivoted to (time × unit) and stacked back to long
form. The spillover represents the neighbour-weighted average of the same
feature at the same lag — capturing how the cercle's neighbourhood context
affects local displacement.

---

## Part I — Linear Panel Regression

### Stage A — Screen: which features have any individual signal?

We fit each of the 282 candidate features alone (with controls and FE) and
rank by |t|-statistic, applying a Benjamini-Hochberg correction at q < 0.10:

$$y_{it} = \beta\, f(x^{(k)}_{it}) + \delta' C_{it} + \alpha_i + \gamma_t + \varepsilon_{it}, \qquad k = 1,\ldots,282$$

**Result.** Four features pass — all conflict spillovers. Zero climate, zero
food pass individually:

| Rank | Feature | coef | t-stat | q(BH) |
|---|---|---|---|---|
| 1 | `W·fatalities_7_9m` | +0.362 | 3.42 | 0.082 ✅ |
| 2 | `W·events_7_9m` | +0.762 | 3.40 | 0.082 ✅ |
| 3 | `W·fatalities_10_12m` | +0.302 | 3.33 | 0.082 ✅ |
| 4 | `W·fatalities_1_3m` | +0.281 | 3.25 | 0.084 ✅ |
| 5 | `W·events_4_6m` | +0.674 | 3.11 | 0.102 ✗ |
| 6 | `sm1_anom_1m_1_3m` *(best climate)* | +6.39 | 3.00 | 0.102 ✗ |
| 7 | `spi6_19_24m` *(best SPI, tied)* | −0.088 | 3.00 | 0.102 ✗ |

The first four are all conflict **spillovers** (`W·X`) — violence in
neighbouring cercles predicts local IDP changes. Climate just misses the
correction (two features tied at q = 0.102). Food appears nowhere near the
top.

Although climate doesn't survive the individual screen, ranks 6–21 are almost
entirely SPI-6 features at 13–18m and 19–24m lags — a coherent cluster just
below the cutoff. This tells us climate has real signal that is simply weaker
than conflict individually, and that the lag windows worth testing are 13–18m
and 19–24m.

<!-- > **Why not stop here?** The screen tests features in isolation. Because conflict
> and climate are correlated, a climate feature might look significant only
> because it proxies for conflict — or fail to appear significant only because
> conflict absorbs its variance. We need a joint model. -->

---

### Stage B — Form: local or spillover?

The screen shows conflict spillovers dominate, but doesn't tell us whether
conflict and climate should enter as **local** terms ($x$, the cercle's own
value) or **spillover** terms ($Wx$, neighbours' value). We test this with
the smallest joint model — one conflict + one climate/food partner, each in
exactly one form:

$$y_{it} = \beta\, f(\text{conflict form}) + \theta\, f(\text{partner form}) + \delta' C_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$$

Fitting all four local/spillover combinations across 1,064 (conflict × partner)
pairs:

| 2-term model | median R²_within | wins (of 1,064 pairs) |
|---|---|---|
| local $x$ + local $z$ | 0.0615 | 184 |
| **spillover $Wx$ + local $z$** | **0.0732** | **564** |
| local $x$ + spillover $Wz$ | 0.0589 | 68 |
| spillover $Wx$ + spillover $Wz$ | 0.0714 | 248 |

"Wins" means: for each pair, which of the four combinations has the highest
R²? `Wx + z` wins 564 of the 1,064 head-to-head contests. The two
spillover-conflict combinations together win 812 (76%).

**Conclusion from Stage B.** Conflict belongs as a **spillover** term; its
partner belongs as a **local** term. This is the one question Stage B can
answer: *in the minimal 2-term model*, which form of each driver carries the
signal. It cannot tell us whether climate spillovers matter once conflict is
fully controlled — that requires a larger joint model. We carry both local and
spillover forms forward into Stages C and D, where the full picture emerges.

---

### Stage C — Variable: which conflict anchor?

Stage B established the form. Stage C asks which conflict **variable** (which
indicator and lag band) anchors the model. We now carry both forms of both
drivers — the four-term pair — and sweep all 8 conflict × 133 climate/food
combinations (1,064 pairs):

$$y_{it} = \beta_1 f(X) + \beta_2 f(WX) + \theta_1 f(Z) + \theta_2 f(WZ) + \delta' C_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$$

Ranking by full-model R², the top five pairs are:

| Rank | Conflict (X) | Partner (Z) | R²_full | local coef (p) | W·X coef (p) |
|---|---|---|---|---|---|
| 1 | `fatalities_10_12m` | `ndvi_anom_19_24m` | 0.098 | +0.035 (0.074) | +0.292 (0.002) |
| 2 | `fatalities_10_12m` | `ndvi_annual_change_19_24m` | 0.095 | +0.036 (0.066) | +0.296 (0.001) |
| 3 | `fatalities_10_12m` | `spi6_13_18m` | 0.095 | +0.036 (0.072) | +0.304 (0.001) |
| 4 | `fatalities_10_12m` | `t2m_anom_7_9m` | 0.094 | +0.037 (0.064) | +0.309 (0.001) |
| 5 | `fatalities_10_12m` | `sm1_anom_4_6m` | 0.094 | +0.036 (0.069) | +0.304 (0.001) |
| **550** | `events_7_9m` *(best events pair)* | `ndvi_anom_19_24m` | 0.069 | +0.103 (0.016) | +0.727 (0.001) |

Three things are immediately clear from the table. First, **`fatalities_10_12m`
appears in every top pair** — whatever climate partner it is matched with, the
conflict anchor is always the same variable (19 of 20 top pairs). Second, the
**W·X spillover coefficient is the stable, dominant signal** (+0.29–0.31*** across
all top pairs), consistent with Stage B's finding that conflict acts through
neighbours. Third, local conflict is also significant (p ≈ 0.06–0.07) but
smaller — both forms are real.

The last row illustrates why W·X coefficient alone is not a sufficient criterion:
`events_7_9m` has a much larger spillover coefficient (+0.727***) than
`fatalities_10_12m` (+0.29–0.31***), yet its R²_full is only 0.069 vs 0.094–0.098.
This is a **measurement-scale artefact** — one event is a smaller shock than one
log-fatality, so the coefficient must be larger to produce the same displacement
effect. R²_within is scale-invariant and correctly identifies `fatalities_10_12m`
as the better anchor.

**No food partner rivals conflict.** Best food-partner R²_full = 0.089 vs
climate-partner best = 0.098; only 7% of food partners are significant in the
full spec.

---

### Stage D — Exhaustive search: lock the backbone, optimal lags, and find M1

Stages B–C suggested `fatalities` as the conflict variable and indicated food
adds nothing. To avoid hand-picking, we confirm this **exhaustively**. We also
remove an unjustified assumption from the simpler pairwise searches: there is
no reason the local and spillover conflict terms must use the same lag band.
Local conflict captures violence in the focal cercle itself — people flee and
register quickly (~7–9m). Spillover conflict captures arrivals from neighbouring
cercles — a slower pipeline through DTM registration (~10–12m). We search over
**independent** local lag $\ell$ and spillover lag $s$:

$$y_{it} = \underbrace{\beta_1 f(\text{conf}_{\ell}) + \beta_2 f(W\!\cdot\!\text{conf}_{s})}_{\text{conflict}} + \underbrace{\theta_1 f(z_1) + \theta_2 f(Wz_1) + \theta_3 f(z_2) + \theta_4 f(Wz_2)}_{\text{two climate variables}} + \underbrace{\phi_1 f(\text{food}) + \phi_2 f(W\!\cdot\!\text{food})}_{\text{food}} + \delta' C_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$$

**Why two climate variables?** One slot forces a single pathway. Climate may
act through distinct mechanisms at different lags (e.g. wet anomaly at 13–18m
vs dry anomaly at 19–24m). Two slots from different physical source series let
the search find the best-fitting independent pair.

**3,951,360 models** (2 conflict types × 4 local lags × 4 spillover lags ×
21 food × 5,880 climate pairs), run on a compute cluster (~10h on 16 cores).
The top models by R²_within:

| Rank | Type | local lag | spill lag | Food | z1 | z2 | R²_within | n_sig |
|---|---|---|---|---|---|---|---|---|
| 1 | fatalities | **7–9m** | **10–12m** | infl_food_1_3m | spi6_13_18m | spi6_19_24m | **0.1025** | 3 |
| 2 | fatalities | 7–9m | 10–12m | infl_food_1_3m | t2m_7_9m | spi6_19_24m | 0.1023 | 3 |
| 3 | fatalities | 7–9m | 10–12m | infl_food_1_3m | spi6_13_18m | spi6_19_24m | 0.1022 | 3 |
| 4 | fatalities | 7–9m | 10–12m | infl_food_4_6m | spi6_13_18m | spi6_19_24m | 0.1020 | 3 |
| 8 | fatalities | 4–6m | 10–12m | infl_food_1_3m | spi6_13_18m | spi6_19_24m | 0.1017 | 2 |

n_sig = regressors significant at p<0.10 (out of 8).

**Three outcomes read from the table:**

**Outcome 1 — Conflict anchor and lags confirmed.** `fatalities` is in every
top model. The spillover lag is **10–12m in 100% of the top 100 models**
regardless of local-lag choice — the DTM registration delay for arrivals is
consistently ~10–12m. The local optimal lag is **7–9m** — people fleeing their
own cercle register 2–3 months faster. The mixed-lag spec (local 7–9m,
spillover 10–12m) beats the same-lag (10–12m/10–12m) by +0.0027 R².

We also check whether **climate spillovers** benefit from different lags
between local and spillover terms. The answer is no — for all three climate
variables the same-band spillover is already optimal or any alternative
produces marginal / mechanistically unclear gains:

| Climate variable | local band | best spillover band | ΔR² vs same-band |
|---|---|---|---|
| `spi6` | 13–18m | **13–18m (same)** | 0.000 — same band is clearly best |
| `p_anom` | current | 13–18m ** | +0.001 — too small to justify a change |
| `streamflow` | current | current (same) | all ns — no clear winner |

The mixed-lag finding is therefore **specific to conflict**: the physical travel
and registration delay for arrivals from neighbours creates a measurable lag
difference. Climate variables affect livelihoods within the focal cercle
directly; their spillover terms reflect neighbours' climate, which acts on the
same timescale as the local term.

**Outcome 2 — Food local is null; W·food is also dropped.** The local food
term is 0% significant across all top 100 models regardless of which food
variable or lag is used — food contributes nothing as an independent driver.
`inflation_food_19_24m` is retained as a null completeness placeholder (most
spatial variation among food variables, giving positive within-R²).

The food **spillover** term (W·food) is also excluded. `inflation_food_price_index`
is a near-national time series — **90% of its variance is between-time** (common
price shocks absorbed by time FE), leaving little spatial variation for a
spillover to capture. Empirically, `W·inflation_food_19_24m` has p=0.729 and
corr(food, W·food) = +0.83 in the raw data. We checked all 21 food variables:
`c_food_price_index` is the exception where W·food is significant (p≈0.035)
because its negative local–spillover correlation captures food price
*differentials* between cercles — but even with that variable the best R²_full
is only 0.063, far below the inflation-food variants (~0.10). The W·food term
is therefore dropped entirely.

The backbone is now fully locked: **`fatalities_7_9m` (local) +
`W·fatalities_10_12m` (spillover) + `inflation_food_19_24m` (no W·food)**.

**Outcome 3 — M1 and candidate second channels identified.** The top models
by raw R² all have n_sig = 3 — one climate variable typically insignificant
(NDVI noise inflation). Applying the both-significant criterion across the
5,880 backbone rows, sorted by R²_within:

| z1 | z2 | R²_within | z1 coef (p) | z2 coef (p) | within-corr | note |
|---|---|---|---|---|---|---|
| `sm3_7_9m` | `t2m_max_13_18m` | 0.0968 | +2.07 (0.21) | −0.04 (0.49) | — | ✗ raw max — both ns (noise) |
| **`spi6_13_18m`** | **`spi6_flood_dur_13_18m`** | **0.0967** | **+0.134 (0.000)** | **−0.056 (0.001)** | **+0.46** | **→ M1** |
| `spi6_13_18m` | `spi6_flood_severity_13_18m` | 0.0960 | +0.131 (0.000) | −0.030 (0.001) | +0.53 | SPI facet pair |
| `t2m_1_3m` | `sm1_7_9m` | 0.0933 | −0.076 (0.048) | +4.407 (0.039) | +0.086 | → M2 candidate |
| `p_anom_cur` | `streamflow_cur` | 0.0914 | −0.002 (0.024) | +0.000 (0.016) | +0.042 | → M3 candidate |

**M1** is the top both-significant pair: `spi6_13_18m` (wet anomaly 13–18m ago
→ displacement, +0.134***) and `spi6_flood_dur_13_18m` (prolonged prior
flooding → wave absorbed, −0.056***). Both coefficients are *** and the
mechanistic story is coherent.

The within-corr of +0.46 between the two SPI terms is a warning sign by the
two-step rule, but both variables survive jointly at p<0.001, and dropping
either collapses the other's coefficient by 60% — confirming they jointly
identify two distinct facets of the same flood signal: a level effect and a
wave-absorption effect. High correlation between flood level and duration is
physically expected (larger floods last longer) and does not invalidate joint
use when both are clearly identified.

**M2 and M3 candidates** are identified from the same table — the best
both-significant pairs that contain no SPI variables (since SPI is already M1)
and have low within-correlation between their two variables:
- **M2** = `t2m_1_3m` + `sm1_7_9m` (R²=0.0933, within-corr=+0.086) — temperature at 1–3m and soil moisture at 7–9m, different lags so near-orthogonal
- **M3** = `p_anom_cur` + `streamflow_cur` (R²=0.0914, within-corr=+0.042) — contemporaneous rainfall and river flow, near-orthogonal

These will be tested as standalone second channels in Stage E, then combined
with M1 to form M4 and M5. **M0** (backbone only, R²=0.087) and **M1**
(backbone + SPI, R²=0.097) are the two reference models from Stage D. Stage E
asks whether a second climate channel adds independent signal beyond M1.

---

### Stage E — Final model: does a second climate channel help?

The Stage D formula has exactly two climate slots ($z_1, z_2$). Stage E asks
whether a **second independent climate channel** adds signal beyond M1. This
requires a new, larger formula that allows two climate families — each with
local and spillover terms:

$$y_{it} = \underbrace{\beta_1 f(\text{conf}) + \beta_2 f(W\!\cdot\!\text{conf})}_{\text{conflict}} + \underbrace{\theta_1 f(c_1) + \theta_2 f(Wc_1) + \theta_3 f(c_2) + \theta_4 f(Wc_2)}_{\text{channel 1}} + \underbrace{\theta_5 f(c_3) + \theta_6 f(Wc_3) + \theta_7 f(c_4) + \theta_8 f(Wc_4)}_{\text{channel 2 (new)}} + \phi\, f(\text{food}) + \delta' C_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$$

where channel 1 is fixed to the Stage D winner (SPI: $c_1$=`spi6_13_18m`,
$c_2$=`spi6_flood_dur`) and channel 2 is the candidate second channel. We
also test standalone alternatives (M2, M3) and combinations (M4, M5). All
models share the locked backbone; M0 and M1 use the Stage D formula (8 terms),
M2–M5 use this extended formula (12 terms):

| Label | composition | climate block | # terms | R²_within | ΔR² vs M1 |
|---|---|---|---|---|---|
| **M0** | backbone only | none | 0 | 0.091 | — |
| **M1** | backbone + Ch1 | SPI {`spi6_13_18m`, `spi6_flood_dur`} | 4 | 0.095 | (base) |
| **M2** | backbone + Ch2a | `t2m_1_3m` + `sm1_7_9m` | 4 | 0.099 | +0.004 |
| **M3** | backbone + Ch2b | water shock {`p_anom_cur`, `streamflow_cur`} | 4 | 0.094 | −0.001 |
| **M4** | M1 + M2 (Ch1 + Ch2a) | SPI + temperature/soil | 8 | 0.102 | +0.007 |
| **M5** | **M1 + M3 (Ch1 + Ch2b)** | **SPI + water shock** | 8 | **0.099** | **+0.004** |

M2 (temperature + soil moisture) and M3 (water shock) are each tested as
standalone second channels alongside M1. M2 alone (R²=0.099, ΔR²=+0.004)
actually fits better than M3 alone (R²=0.094, ΔR²=−0.001). But the two-channel
combinations tell a different story: in **M4** (SPI + M2), `t2m_1_3m` drops to
**ns (p=0.127)** in the joint model — the temperature signal is partly absorbed
by the SPI channel when both are present, and only the soil moisture component
survives. In **M5** (SPI + M3), all four climate variables remain significant in
the joint model (p_anom p=0.023**, streamflow p=0.020**). M5 therefore provides
a cleaner two-channel decomposition: one slow SPI channel and one contemporaneous
water-shock channel, both independently identified. **M5 is the recommended
specification.**


### The final model: M5

$$y_{it} = \underbrace{\beta_1 f(\text{fat}_{7\text{-}9}) + \beta_2 f(W\!\cdot\!\text{fat}_{10\text{-}12})}_{\text{conflict (mixed lag)}} + \underbrace{\theta_1 f(\text{spi6}_{13\text{-}18}) + \theta_2 f(\text{flood\_dur}_{13\text{-}18}) + \theta_3 f(W\!\cdot\!\text{spi6}) + \theta_4 f(W\!\cdot\!\text{flood\_dur})}_{\text{SPI flood channel}} + \underbrace{\theta_5 f(\text{p\_anom}) + \theta_6 f(\text{stream}) + \theta_7 f(W\!\cdot\!\text{p\_anom}) + \theta_8 f(W\!\cdot\!\text{stream})}_{\text{acute water channel}} + \phi\, f(\text{food}_{19\text{-}24}) + \delta' C_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$$

**R²_within = 0.0992**, n = 1,417, 44 cercles, SE clustered on entity:

| Driver | Coef | SE | p | |
|---|---|---|---|---|
| log1p_idp_t1 (AR control) | −0.179 | 0.045 | 0.000 | *** |
| fatalities_7_9m (local) | +0.046 | 0.022 | 0.032 | ** |
| **W·fatalities_10_12m (spillover)** | **+0.252** | **0.090** | **0.005** | *** |
| spi6_13_18m | +0.139 | 0.034 | 0.000 | *** |
| spi6_flood_dur_13_18m | −0.054 | 0.014 | 0.000 | *** |
| W·spi6_13_18m | −0.311 | 0.154 | 0.043 | ** |
| W·spi6_flood_dur_13_18m | +0.120 | 0.094 | 0.202 | ns |
| p_anom_1m (current) | −0.002 | 0.001 | 0.023 | ** |
| streamflow_1m (current) | +0.0001 | 0.00003 | 0.020 | ** |
| W·p_anom_1m | +0.004 | 0.005 | 0.417 | ns |
| W·streamflow_1m | −0.000 | 0.000 | 0.839 | ns |
| inflation_food_19_24m | +0.005 | 0.003 | 0.122 | ns |

**Interpreting M5.** The AR control `log1p_idp_t1` (−0.179***) is the dominant
single term — cercles with high IDP stock grow more slowly (mean reversion). It
is not an incidental nuisance but a structural feature of the data: IDP counts
are highly persistent, so last period's stock strongly predicts this period's
change direction. Without this control, fitting conflict + climate + food alone
gives R²_within < 0 — the model predicts *worse* than the fixed-effects baseline
alone, because the uncontrolled mean-reversion structure overwhelms the driver
signals. Including `log1p_idp_t1` corrects for this and gives all drivers a fair
chance to explain the residual variation.

The cleanest way to see what the drivers contribute is to split the model in
two steps. **Step 1:** fit only the controls (AR + `months_between`) — this
gives R²_within = 0.049, capturing the mean-reversion structure alone.
**Step 2:** add the three driver families on top — this adds +0.050, bringing
the total to R²=0.099. That +0.050 is what conflict, climate, and food
collectively explain *beyond* the auto-correlation structure.

Within that +0.050, a **LOO (Leave-One-Out) decomposition** — refit with each
driver block removed, measure the R² drop — shows the relative importance:

| Driver block | ΔR² when removed | share of driver contribution |
|---|---|---|
| **Conflict** (local + spillover) | −0.037 | **75%** |
| **Climate** (all 8 terms) | −0.008 | **16%** |
| Food | +0.005 | — (null; marginally hurts fit) |

Conflict accounts for three-quarters of the driver-explained variance; climate
adds a further 16%. Food is null — its coefficient is near zero and removing it
marginally *improves* fit, which is why it is kept only as an explicit null
finding rather than a genuine contributor.

Conflict drives displacement through **neighbours** more than through the focal
cercle (spillover +0.252*** vs local +0.046**). The 5× ratio is consistent with
people fleeing *toward* safer cercles — registering as IDP arrivals there rather
than as departures from the conflict cercle.

**SPI flood channel** — this channel operates with a 13–18 month lag and
captures two distinct effects of past rainfall on the same variable:

- `spi6_13_18m` (+0.139***): a **wet anomaly** 13–18 months ago increases
  current displacement. The mechanism is flooding — above-normal rainfall
  causes crop damage, home destruction, and health risks that push people out
  over the following months.
- `spi6_flood_dur_13_18m` (−0.054***): a **longer flood duration** at the same
  lag *reduces* current displacement. This is the wave-absorption effect — if
  the flooding was prolonged, the vulnerable population has already fled in
  earlier periods, leaving fewer people to displace now.

Together the two terms say: a sudden, short-lived wet shock drives a large
displacement wave; a prolonged flood gradually drains the catchment population
so that later displacement is smaller. The net effect on any given observation
depends on both the level and duration of the past wet anomaly.

The **spillover `W·spi6_13_18m`** (−0.311**) is negative and larger in
absolute magnitude than the local term. This means a flood anomaly in
*neighbouring* cercles *reduces* local IDP counts. The mechanism is spatial
absorption: when neighbours flood, their displaced population flows *out* toward
less-affected cercles (including the focal cercle). The focal cercle receives
arrivals, but these arrivals are already counted in its IDP stock through the
IDP-receipt pipeline. The negative spillover captures a different dynamic —
when neighbours have high SPI, there is less net *outward* pressure from the
focal cercle itself because neighbours' floods are directing people away, not
toward conflict zones. Both signs are empirically robust and mechanistically
coherent.

**Acute water channel** — unlike SPI which captures rainfall accumulated over
months, this channel operates in the **current observation window**:

- `p_anom_1m` (−0.002**): current **rainfall is protective**. Higher rainfall
  during the observation period reduces displacement — rainfall supports
  agricultural livelihoods, reducing food stress and the need to migrate. The
  negative sign reflects a contemporaneous stabilising effect.
- `streamflow_anom_1m` (+0.0001**): current **river flow is displacing**. Higher
  streamflow means active flooding of riverbanks and floodplains, directly
  forcing displacement. The coefficient is tiny in absolute terms but the
  variable has a wide range — a one-standard-deviation increase in streamflow
  (≈ +85 m³/s above normal) corresponds to roughly +0.03 additional monthly
  log-IDP growth, comparable to one-tenth of the conflict spillover effect.

The opposite signs of rainfall (protective) and streamflow (displacing) in the
same current window are not contradictory — they capture two different aspects
of water: rainfall on land supports crops, while river flooding destroys
them. Both are robustly significant across specifications and lag choices.

Food (inflation_food_19_24m) is null (p=0.122). This is not a modelling failure
— it is a finding. All 21 food variables, at all lag bands, are null in the
multivariate setting. Food's apparent signal in raw correlations is a
cross-sectional baseline artefact absorbed by entity FE.

---

## Part II - Random Forest
### Is M5 misspecified? Random Forest validation and functional-form tests

### How we use Random Forest

Panel regression with two-way fixed effects (**TWFE**) and Random Forest (**RF**) are fundamentally different models, and that
difference is precisely what makes RF useful as a complement.

**TWFE** imposes strong structure: linearity, additivity, and — crucially —
**fixed effects** that absorb all time-invariant differences between cercles and
all common time shocks. The coefficients estimate within-cercle, within-time
associations. The price of this identification structure is that it can only
fit linear, additive relationships, and it discards the cross-sectional
baseline information that entity FE removes.

**Random Forest** imposes none of these constraints. It fits splits and branches
that can capture thresholds (a variable only matters above a certain level),
interactions (the effect of conflict depends on drought severity), and non-linear
curves — all without assuming linearity or additivity. Crucially, it does **not**
assume fixed effects: when run on the raw (un-demeaned) data it retains the
cross-sectional baseline differences between cercles that TWFE discards. This
gives RF access to more structure, but at the cost of causal identification —
RF has no standard errors, no FE, and no way to separate genuine causal signals
from confounding by stable cercle characteristics.

We exploit this difference in two ways:

1. **Misspecification diagnostic.** We fit RF on the *within-transformed* M5
   features (entity + time demeaned — the same variation that TWFE uses) and
   compute non-linearity scores and Friedman H-statistics for interactions.
   If RF finds that a feature's contribution is highly non-linear, it is
   suggesting that TWFE's linear coefficient may be missing real structure.
   Every such signal is then **tested formally in TWFE** — the only framework
   that provides credible inference.

2. **Predictive validation.** We evaluate RF out-of-sample using **entity-blocked
   5-fold cross-validation (OOS R²)** — "OOS" means *out-of-sample*, i.e. the
   model is fit on some cercles and evaluated on held-out cercles it has never
   seen. This prevents overfitting and gives an honest estimate of how well the
   model generalises. The M5-restricted RF (14 features) achieves OOS R² = 0.089
   — *better* than the full 282-feature RF (0.075). Adding 268 more features
   hurts generalisation, confirming that M5's variable selection is not just
   good for in-sample fit but also predictively sound.

The raw (un-demeaned) RF ceiling is OOS R² = 0.144. The gap between 0.144 and
TWFE's 0.099 reflects the cross-sectional baseline structure (stable cercle
differences) that entity FE removes by design — not a specification failure on
TWFE's part. The two methods are answering different questions: TWFE asks *what
drives within-cercle change*; raw RF asks *what predicts displacement levels
overall*.

Even at its best (OOS R² = 0.144), RF is **not good enough to build a
reliable operational forecast** — 86% of the variance in displacement remains
unexplained. Displacement is driven by many factors beyond the three driver
families we measure (local governance, inter-community relations, individual
household decisions) and the data we have cannot capture them. This means RF's
role here is not to provide predictions, but to serve two narrower purposes:
**cross-checking** that M5's variable selection is sound (the M5-restricted RF
generalises better than the full-feature RF, confirming the TWFE funnel
selected the right variables), and **functional-form testing** (flagging
non-linearities and interactions for TWFE to verify). Both uses are valuable
precisely because they do not require RF to be a good predictor overall.

### RF configuration and results

**What we feed into RF.** We fit RF on the **within-transformed** M5 features —
the same entity + time demeaned variation that TWFE uses (subtract the
cercle mean, the time mean, and add back the grand mean for each variable).
This keeps RF in the same identification space as TWFE, so that any pattern RF
finds is a pattern in the within-cercle variation, not in stable baseline
differences between cercles. The RF uses 500 trees, `max_features=0.2`,
entity-blocked 5-fold cross-validation (folds defined by cercle, so each
held-out fold is an unseen set of cercles). Permutation importance is computed
on the held-out test fold of each CV split — not on the training data — which
prevents inflated importance scores.

**OOS R² results:**

| RF variant | features | OOS R² |
|---|---|---|
| Full demeaned (282 features) | all candidate features, within-transformed | 0.075 ± 0.066 |
| **M5-restricted demeaned** | 14 M5 features, within-transformed | **0.089 ± 0.041** |
| Full raw (282 features) | raw features + entity OHE + month index | 0.144 ± 0.061 |
| M5 raw | 14 M5 features, raw | 0.121 ± 0.043 |

M5-restricted RF achieves better OOS R² than the full 282-feature RF (0.089 vs
0.075) — adding 268 more features actively hurts generalisation, confirming
that TWFE's variable selection is predictively sound.

**Non-linearity scores.** For each M5 feature we compute a non-linearity score —
how much of its RF predictive contribution comes from non-linear splits vs a
linear approximation. A score near 1.0 means the RF is heavily exploiting
non-linear structure; near 0 means the feature is essentially linear:

| Feature | nl score | what this flags |
|---|---|---|
| `W·streamflow_cur` | 0.915 | highly non-linear |
| `fatalities_10_12m` | 0.913 | highly non-linear |
| `streamflow_cur` | 0.826 | highly non-linear |
| `spi6_flood_dur_13_18m` | 0.484 | moderate |
| `p_anom_cur` | 0.397 | moderate |
| `spi6_13_18m` | 0.025 | essentially linear |

**Top interaction H-statistics** (Friedman H — how much the joint partial
dependence of two features exceeds the sum of their individual effects):

| Feature 1 | Feature 2 | H-stat | type |
|---|---|---|---|
| `fatalities_10_12m` | `spi6_flood_dur_13_18m` | 0.409 | conflict × climate |
| `fatalities_10_12m` | `p_anom_1m` | 0.276 | conflict × climate |
| `fatalities_10_12m` | `spi6_13_18m` | 0.211 | conflict × climate |

### What we test in TWFE and why

The non-linearity score tells us *that* a feature's response is non-linear,
but not *what form* the non-linearity takes. For this we inspect the **1D
Partial Dependence Plot (PDP)** — the RF's estimated average response of the
outcome to each feature, holding others fixed. The PDP shape guides which
TWFE test to run:

- **Step-function PDP** (flat then jumps at a threshold) → test a **threshold
  dummy** `I(x > pN)` in TWFE, either replacing or alongside the continuous term
- **Curved PDP** (monotone but bent) → test a **quadratic term** `x²`
- **PDP that flattens only when another feature is high** → test a **product
  interaction** `x × z`

This means we do **not** blindly run all three tests for every flagged feature.
The PDP pre-selects both *which feature* to test and *which functional form*
to try, reducing the number of TWFE tests to those with a clear visual
motivation. The H-statistic from 2D PDPs similarly nominates the most promising
interaction pairs rather than testing all combinations.

Every nominated test is then run in TWFE with two-way FE and clustered SE —
the RF signal is only a nomination, the TWFE result is the verdict.

### Functional-form tests (M6–M10)

The RF flagged five candidates for TWFE testing — three from high nl_scores,
one from a top H-statistic, and one additional test motivated by the Stage A
screen (the spi6_19_24m drought signal that narrowly missed BH correction):

| Test | RF motivation | TWFE test form | TWFE verdict |
|---|---|---|---|
| **M6** spi6 19–24m drought | Stage A rank 7 (nl=0.025, not RF-flagged) | add `spi6_19_24m` as second SPI lag | ❌ significant alone but R²_within drops |
| **M7** streamflow threshold | nl=0.826 — step-function PDP | `I(stream > p75)` dummy ± continuous | ✅ threshold ** displaces continuous term |
| **M8** fatal × flood-dur interaction | H=0.409, top conflict×climate pair | product term `fatal × spi6_flood_dur` | ❌ all ns with updated M5 (was * in old same-lag spec) |
| **M9** conflict non-linearity | nl=0.913 — highly non-linear PDP | quadratic `[log1p(fat)]²` + threshold | ❌ all null — log1p transform artefact |
| **M10** food non-linearity | high raw-RF rank (not demeaned RF) | quadratic, threshold, interaction, shorter lag | ❌ all null — entity FE absorbs |

We test **all RF-flagged candidates** — there is no further pre-selection beyond
what the PDP shape already determined. The second H-statistic pair
(`fatal × p_anom`, H=0.276) was subsumed within M8's joint specification.

**M6** was not RF-flagged (nl_score=0.025, essentially linear) but tested
because the Stage A screen showed `spi6_19_24m` narrowly missed the BH
correction with a clearly negative coefficient. This is an example of using
domain knowledge alongside RF to nominate tests.

Two tests revealed artefacts rather than real structure. **M9**: fatalities had
the second-highest nl_score (0.913), suggesting a strongly non-linear response.
In TWFE, all non-linearity tests are null — the RF was detecting the curvature
of the log1p transform applied to fatalities before regression, not a real
causal curve. After log1p, conflict is genuinely linear. **M10**: food appeared
important in the raw (undemeaned) RF but not in the demeaned RF — the apparent
signal reflects stable cross-cercle differences in food prices that entity FE
absorbs entirely, exactly confirming the food-null result from M5.

**Sub-model labels.** Each functional-form test (M7–M10) runs several
variants, labelled M7a, M7b, … to distinguish them:

| Label | M7 (streamflow) | M8 (interaction) |
|---|---|---|
| **a** | M5 + `fatal × streamflow` product | M5 + `fatal × flood_dur` (local×local) |
| **b** | M5 with continuous stream *replaced* by `I(stream>p75)` | M5 + `W·fatal × flood_dur` (spillover×local) |
| **c** | M5 + `fatal × I(stream>p75)` | M5 + `fatal × W·flood_dur` (local×W·spillover) |
| **d** | M5 + `I(stream>p75)` *alongside* continuous | M5 + all three jointly |

**M7 — streamflow upper-tail threshold.** The threshold dummy is added as an
**extra term alongside** the continuous streamflow (M7d) — not replacing it. This
tests whether the threshold captures *additional* structure beyond the linear
gradient. We also test replacing the continuous term (M7b) to understand where
the signal lives:

| Spec | R²_within | ΔR² | I(stream>p75) | streamflow (continuous) |
|---|---|---|---|---|
| M5 | 0.0992 | — | — | +0.0001** |
| M7b: threshold *replaces* continuous | 0.0985 | −0.0007 | +0.115** | — |
| M7d: threshold *alongside* continuous | **0.1004** | **+0.0013** | **+0.099*** | **ns** |

M7b is *worse* than M5 — replacing the continuous term loses information about
the full distribution. M7d adds +0.001 R²: the threshold captures the upper-tail
jump (+0.099*), while **the continuous streamflow falls to ns**. This is the
key finding — the continuous term's ** significance in M5 was driven almost
entirely by extreme high-flow events, not by the full linear gradient. The
non-linearity is real, but modest.

Critically, **all other M5 coefficients are essentially unchanged in M7d**:

| Driver | M5 | M7d |
|---|---|---|
| fatalities_7_9m | +0.046** | +0.046** |
| W·fatalities_10_12m | +0.252*** | +0.250*** |
| spi6_13_18m | +0.139*** | +0.140*** |
| spi6_flood_dur | −0.054*** | −0.053*** |
| p_anom | −0.002** | −0.003** |

M5 retains the continuous term for parsimony — the threshold adds only +0.001
R² and weakens (but does not eliminate) the continuous term. The overall
conclusion is unchanged.

**M8 — conflict × flood-duration interaction.** Three product terms are added
jointly (local×local `fatal×flood_dur`, W·conflict×local, local×W·flood_dur):

| Spec | R²_within | ΔR² | fatal×flood_dur | W·fat×flood_dur | fat×W·flood_dur |
|---|---|---|---|---|---|
| M5 | 0.0992 | — | — | — | — |
| M8a: LL alone | 0.0992 | +0.000 | ns | — | — |
| M8d: all three jointly | 0.1017 | +0.0026 | ns | ns | ns |

With the updated M5 (local conflict at 7–9m), **all three interaction terms are
insignificant**. The ΔR²=+0.003 in M8d reflects the joint model absorbing some
shared variance, not any individually significant interaction. Importantly, the
original M5 linear coefficients remain stable in M8d — conflict and SPI change
by less than 10%:

| Driver | M5 | M8d |
|---|---|---|
| fatalities_7_9m | +0.046** | +0.052*** |
| W·fatalities_10_12m | +0.252*** | +0.256*** |
| spi6_13_18m | +0.139*** | +0.138*** |
| spi6_flood_dur | −0.054*** | −0.059*** |

**M5's additive linear specification holds.** Adding the threshold (M7d) or
interactions (M8d) leaves the substantive M5 findings — conflict dominance,
SPI wave-absorption, acute water channels — essentially unchanged.

**Summary: RF stress-tested M5 and mostly validated it.** One real threshold
(streamflow), one weak interaction (conflict × flood-duration), several
artefacts dismissed.

---

## Part III — What M5 averages away: heterogeneity

TBD

## Limitations and open questions

TBD