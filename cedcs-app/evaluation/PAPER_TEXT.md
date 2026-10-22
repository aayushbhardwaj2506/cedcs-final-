# Paper text drafted from the evaluation results

Everything below uses numbers from `evaluation/results/RESULTS.md` (seed 42; 20 synthetic worlds x 100 cases = 2,000 cases per
condition). Read `METHODS.md` first: all effectiveness numbers describe a simulation. Tables and figures to insert are in
`evaluation/results/` (CSV files and `figures/*.png`).

---

## Changes your current draft needs (before the new sections)

1. **Section 6.1 (Execution Status) is out of date.** The AI multi-agent layer, live maps and hospital discovery, e-mail dispatch,
   the hospital portal, saved run history, an analytics dashboard and photo/video scene analysis are now implemented and tested
   (more than 300 automated tests). Replace "design stage" accordingly, and state that the E1-E6 experiments *have* been run, on a
   synthetic world.
2. **The AI layer is not "a local large language model".** It uses hosted models (Groq and NVIDIA) with automatic failover. Reword Section 3.1 (L3).
3. **"Immutable, hash-chained audit log" (L7) is not implemented.** The system keeps a complete, replayable per-run audit trail and stores
   every run, but the entries are not hash-chained. Either say "append-only audit trail" or add the chaining.
4. **Voice input, five role-specific consoles, hospital HIS/FHIR feeds and blood-bank network adapters (L1, L6) are not built.** The system has an
   operator screen, a hospital portal, a live map, and discovers real hospitals from OpenStreetMap (capacity unknown unless a hospital
   reports it). Describe them as future work.
5. **Contribution C1 must be reworded** (see 5.4). The data do not show that freshness modelling reduces wrong recommendations; they show it
   makes the stated confidence honest.
6. **The Introduction says Section 5 "reports the results"**: it now will. Update the roadmap sentence, and move the worked example to 5.1.

---

## 5. Results

### 5.1 Setup (one paragraph)

We evaluated the decision core of CEDCS (the project's own freshness scoring, eligibility gate, priority-dependent ranking and
confidence code) against four baselines and a language-model baseline in a synthetic city: 20 independently generated worlds of 60 hospitals
(five archetypes), each with a hidden true state at decision time and a report history that ages, and 2,000 cases drawn from eight clinical
templates with a ground-truth requirement profile. A recommendation is *appropriate* if the chosen hospital truly has every REQUIRED
capability; a recommendation that is not appropriate stands in for an unplanned secondary transfer (*undertriage*). We report three city
loads (relaxed, normal, crowded; *normal* was fixed as the headline before any results were seen). Intervals are 95 % (Wilson); policies are
compared pairwise on identical cases with an exact McNemar test. Because thousands of simulated cases make almost any difference
statistically significant, we emphasise effect sizes. All rates in the simulation are assumptions (Section 6.2), not estimates from data.

### 5.2 E1: main comparison

**Table 6.** Normal city load, 2,000 cases per policy.

| Policy | Undertriage (recommendation misses a REQUIRED capability) | Safety violations (reported data already showed a gap) | Overtriage | Excess ETA, min | Cost-weighted error |
|---|---|---|---|---|---|
| B1 nearest hospital | 53.4 % [51.3, 55.6] | 1,034 (51.7 %) | 0.0 % | 0.0 | 5.34 |
| B2 nearest with an emergency department | 42.1 % [40.0, 44.3] | 811 (40.6 %) | 1.1 % | 0.3 | 4.24 |
| B3 distance + specialty filter | 14.9 % [13.5, 16.6] | 300 (15.0 %) | 1.7 % | 0.8 | 1.55 |
| B4 soft multi-criteria ranking, no gate, no freshness | 15.3 % [13.8, 16.9] | 281 (14.1 %) | 3.1 % | 1.7 | 1.64 |
| **CEDCS** | **0.9 % [0.6, 1.5]** | **0 (0.0 %)** | 4.0 % | 3.6 | **0.32** |

CEDCS recommended a hospital lacking a required capability in 0.9 % of cases, against 14.9 % for the best simple baseline and 53.4 % for
nearest-hospital routing (p < 0.001 against each). The gain is larger when the city is busy: in the *crowded* condition CEDCS's undertriage
is 3.9 % [3.1, 4.8] against 22.4 % for the soft ranking and 28.7 % for the specialty filter; in the *relaxed* condition it is 0.2 % against
11.8 % to 12.4 %. The price is a longer trip (3.6 minutes more than the nearest appropriate hospital, against 0.0 to 1.7 for the baselines) and
slightly more over-triage (4.0 % against 1.7 % to 3.1 %); the cost-weighted error, which weights a missed capability much more heavily than a
longer trip (Section 4.3), is 0.32 for CEDCS against 1.55 to 5.34.

*Where it still fails.* In the crowded city, undertriage of HIGH and CRITICAL cases is 7.9 % and 5.1 % (soft ranking: 28.9 % and 31.1 %). The
residual errors are cases where a hospital *reported* a free ICU bed or a working scanner that was no longer true by decision time.

**A language model without a gate (B5).** On 60 cases, a language model that read the patient text and the 12 nearest hospitals' reported capabilities
and picked one chose a hospital lacking a required capability in 30.0 % [19.9, 42.5] of cases (15 of 60 recommendations were to a hospital
whose own report already showed the gap), against 0 of 60 for CEDCS on the same cases. The interval is wide because of the small
subset; the direction supports the paper's design principle that a model may describe a decision but should not make it.

**Confidence.** CEDCS's stated confidence was informative at the top: recommendations labelled HIGH were truly appropriate 100 % [98.3, 100]
of the time (n = 227, normal load) and 99.5 % in the crowded city, against 98.9 % and 95.7 % for MODERATE. The LOW label did not
single out errors (20 to 24 cases per load, 95.8 % to 100 % appropriate), so we do not claim it is discriminative.

### 5.3 E3: the safety property under wrong triage

We injected triage errors (a required capability dropped and the priority lowered) into 0 % to 50 % of cases. In every condition the gate
never recommended a hospital whose reported data showed that a capability of the profile it was *given* was absent (0 of 2,000 at each of the
five error rates, 10,000 recommendations in all); the soft ranking did so in 14 % to 15 % (281 to 297 of 2,000). This is the property the
gate guarantees by construction, and the experiment confirms the implementation. It is a guarantee about the profile the gate receives, not about the
patient: when the triage itself was wrong, the share of recommendations that missed a capability the patient truly needed was 11 % to 14 %
for CEDCS, no better than the specialty filter (14 % to 15 %) and less than half the soft ranking's rate (25 % to 30 %). An eligibility gate cannot compensate for a
wrong requirement profile; that is the job of the triage stage, the critic and the red-flag rules.

### 5.4 E2: freshness (contribution C1, revised)

We varied the age of every reported value from 0 to 12 hours. Undertriage rose with age for every policy (CEDCS with freshness: 0.1 %, 0.8 %,
2.5 %, 5.6 % at 0, 1, 4, 12 hours; soft ranking: 10.7 %, 15.6 %, 16.9 %, 25.6 %), but **modelling freshness did not reduce it**: with the
freshness model switched off, CEDCS made slightly fewer wrong recommendations (0.7 %, 2.2 %, 4.8 % at 1, 4, 12 hours; the difference is
significant only at 12 hours, p = 0.007, in the direction *against* the freshness model). What freshness changes is what the system *claims to know*.
With the model off, CEDCS stated HIGH confidence in 1,470 of 2,000 cases when the data were 12 hours old, and 95.4 % of those were
right; with it on, CEDCS stated HIGH confidence in only 2 cases (all right). We therefore reword the contribution: freshness
modelling does not improve which hospital is chosen, it keeps the stated confidence from overstating what stale data justify.

### 5.5 E5 and E6: robustness and escalation

*Weights (E5).* Over 20 random perturbations of every weight by up to 20 %, undertriage stayed between 0.9 % and 1.0 % and the top choice matched the
baseline weights in 93.5 % to 97.9 % of cases (mean 96.2 %): the ranking is not fragile to the exact values of Table 1.

*Escalation (E6).* We removed each of six key capabilities from each hospital with probability q (0 to 0.95). The radius widening grew from 12.8 % of
cases (q = 0) to 42.5 % (q = 0.6); operator handoff rose from 0 % to 31.6 % (q = 0.95), and **never occurred when a truly appropriate hospital
existed** (0.0 % at every q). No recommendation ever contradicted the reported data (0 violations). Two limits appear: undertriage of the
recommendations that were issued rose from 0.9 % to about 15 % for q ≥ 0.6, and in cases where no appropriate hospital existed CEDCS handed off in 65.5 % to 77.0 % of them at high q and still issued a
recommendation in the other 23 % to 35 %, because hospitals whose capability is unreported are kept as *provisional* rather than rejected. A recommendation made from unknown data is flagged for telephone confirmation, but it is still a recommendation.

### 5.6 Triage under noisy input (E4)

Using the application's *rule-based* triage (the offline fallback, not the language-model triage), recovery of the true REQUIRED capabilities was
low even for clean text (mean recall 0.47 at noise level 0, 0.43 at level 3), and the priority was underestimated in 47 % to 58 % of cases; the
resulting undertriage rose from 17.7 % to 28.0 %. This quantifies why the language-model triage exists; its own accuracy on the same texts is **not**
reported here (a first attempt was cut short by the free-tier daily token limit and is left as future work).

### 5.7 Latency and where it goes

The deterministic decision core takes 0.73 ms per case on average (median 0.48 ms, 95th percentile 1.88 ms; n = 2,000). The full live pipeline, measured on 45
real runs with the language models and map services, had a median of 24.5 s, a 95th percentile of 104.8 s and a maximum of 249 s. The time
is almost entirely in the language-model and map stages: the intake/triage/clarification/critic phase ("front crew") has a median of 22.0 s, of
including resource interpretation at 11.0 s (95th percentile 60.4 s) and facility discovery 4.9 s (36.6 s); the pipeline phase that contains
the decision engine takes a median 1.1 s, which also includes the language-model advisor (1.1 s). The rule-based stages of the pipeline (red flags, eligibility, ranking,
validation) are below 2 ms each. The long tail comes from the external services' free tiers (rate limits, cold starts), not from the decision logic. Of the
282 language-model calls in these runs, 97.9 % succeeded and 6 fell back to rules. In answer to the research question's latency
component: autonomy did not make the decision slow; it moved the time to the interpretation and data-gathering stages.

### 5.8 What this evaluation does and does not show

It shows that, in this simulated city, an eligibility gate over fresh-weighted data reduces recommendations to hospitals lacking a required capability from
15 % to 53 % (normal load) to 0.9 %, and from 22 % to 60 % to 3.9 % in the crowded city, that the gate's safety property holds under triage errors, and that the ranking is stable. It does not show clinical
benefit, real-world prevalence of these failures, or that the simulation's parameters match any real city; it does not evaluate the language-model
triage; and freshness modelling did not improve accuracy.

---

## Replace Section 6.1 with (draft)

*Execution status.* All seven layers of the architecture exist in prototype form except voice input, role-specific consoles for each of
five actors, and live hospital HIS/FHIR integration. The decision core, the AI multi-agent layer (hosted models with automatic failover), live
mapping and real hospital discovery from OpenStreetMap, e-mail dispatch with a hospital confirmation loop, a hospital portal, saved run history
and an analytics dashboard are implemented and covered by more than 300 automated tests. The evaluation in Section 5 was run on a synthetic world
because live, cross-organizational EMS-hospital data was not available; real hospitals' bed and equipment data are not published anywhere, so in
live operation the system treats them as unknown and flags the recommendation for telephone confirmation.

## Data availability (add near the end)

The evaluation code, the synthetic-world generator, all CSV outputs and figures are in the project repository (`cedcs-app/evaluation/`); the
experiments are reproducible with `python -m evaluation.run --worlds 20 --cases 100 --seed 42` (the language-model baseline B5 and the
real-latency table additionally need provider keys and the saved run history).
