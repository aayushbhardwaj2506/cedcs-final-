# Evaluation methods and assumptions

This folder implements the evaluation framework of the paper (Section 4): baselines B1-B5, metrics, and experiments E1-E6, plus a
latency decomposition. It runs the project's **real decision code** (`deterministic_core/`: freshness scoring, eligibility gate,
priority-dependent ranking, confidence, re-verification) on a **synthetic world with a hidden true state**.

Reproduce everything: `python -m evaluation.run --worlds 20 --cases 100 --seed 42 --llm-baseline 60 --real-latency`
(about 10 minutes; B5 and the real-latency table need the API keys and the history database; without those flags the run is fully
deterministic). Tests: `python -m pytest tests/test_evaluation.py`.

## What is real and what is simulated

| Real | Simulated |
|---|---|
| The deterministic core, exactly as the application runs it (the 8 km / 25 km widening, the minimum of 3 eligible, provisional handling of unknown data, freshness decay with the application's time constants and source trust, the priority-dependent weights of Table 1). | The hospitals, their capabilities and beds, the true state at decision time, and how reports age. |
| For B5: a real language model (the application's Groq / NVIDIA failover) choosing a hospital. | The patients: eight clinical templates with a ground-truth requirement profile; text variants written by hand. |
| For the E4 triage: the application's own rule-based triage. | Travel times: haversine distance at 30 km/h x 1.4 (the application's estimate when no routing service is used), treated as exact. |
| For the latency table: the measured stage times of real runs saved by the application. | |

Consequently every effectiveness number is a statement about **behaviour in this simulation**, not about real patients or hospitals.

## The world (`world.py`)

* **Hospitals.** 60 per world in a 30 km square, five archetypes (tertiary 15 %, trauma 10 %, district 30 %, private mid-size 25 %,
  clinic/nursing home 20 %), each with per-capability probabilities (e.g. a district hospital has a cath lab with probability 0, a CT
  scanner 0.4, an ICU 0.6). Bed counts and occupancy are drawn per hospital.
* **City load.** `relaxed`, `normal`, `crowded` multiply the chance that a bed is free by 1.8, 1.0 and 0.4. `normal` was fixed as the headline
  condition before any results were seen; all three are always reported.
* **True state at the decision time T0.** Every resource has a true value (department active, equipment operational, free beds, blood bank,
  specialist on site / on call / none).
* **Reports.** What a policy sees is what a hospital reported earlier. A report was true when made; the resource then kept changing, so
  the report is the true value passed back through random changes: beds move by +-1 at a per-resource rate (ICU 0.25 / h, emergency
  beds 1.5 / h, general 0.5 / h, HDU 0.3 / h, paediatric 0.3 / h, ventilators 0.2 / h), equipment flips at 0.01 / h, departments at
  0.0005 / h, blood bank at 0.03 / h, specialist status is redrawn at 0.2 / h. Each field is unreported (UNKNOWN) with probability 0.10.
  Report age: log-normal, median 45 minutes, about 8 % older than 6 hours (E2 fixes it to 0, 1, 4 or 12 hours, +-20 %). Source:
  hospital console 55 %, government 25 %, FHIR 20 % (the application's source-trust weights apply).
* **Cases.** 30 % CRITICAL, 30 % HIGH, 25 % MODERATE, 15 % LOW, drawn from eight templates (cardiac, neurological, trauma, paediatric,
  obstetric, respiratory, fracture, minor) within 10 km of the centre.

**All rates and probabilities are plausibility choices, not fitted to data.** E2 (staleness), E5 (weights) and the three load levels
exist precisely to show how much the conclusions depend on them. A different world would give different absolute numbers.

## Ground truth and metrics (`metrics.py`)

A recommendation is **appropriate** when every REQUIRED capability of the case's ground-truth profile is truly present at the chosen
hospital at T0. The ground truth is defined on the requirement profile, not on a historical destination label.

* **Undertriage** = share of issued recommendations that are not appropriate (an *unplanned secondary transfer* would follow). Reported over
  all issued recommendations, and over the *avoidable* cases (a truly appropriate hospital existed within 25 km).
* **Handoff** = no recommendation (CEDCS: nothing eligible, a human operator takes over: a planned pathway, not counted as an error).
* **Safety violation** = a recommendation for a hospital whose *reported* data says a REQUIRED capability is absent. By construction 0 for
  the gate; baselines can violate it.
* **Overtriage** = a LOW/MODERATE case sent to a tertiary or trauma centre although a closer, truly appropriate smaller hospital existed.
* **Required-capability coverage** = mean share of REQUIRED capabilities truly present at the chosen hospital.
* **Excess ETA** = ETA of the chosen hospital minus ETA of the nearest truly appropriate one (avoidable cases).
* **Cost-weighted error** = 10 x undertriage + 1 x overtriage + 0.05 x excess minutes (magnitudes assumed; only the ordering
  lambda_under >> lambda_over > lambda_time comes from the paper).
* **Calibration** = share truly appropriate, by the confidence level CEDCS stated.
* Intervals: Wilson 95 % for proportions, bootstrap 95 % (1000 resamples) for means; paired exact McNemar test between policies on the
  same cases. With thousands of simulated cases almost any difference is "significant": read the effect sizes, not the p-values.

## Policies (`policies.py`)

| ID | What it does |
|---|---|
| B1 | nearest hospital |
| B2 | nearest hospital whose report says it has an emergency department |
| B3 | nearest hospital with an emergency department and the case's specialty department (a distance + specialty filter) |
| B4 | the CEDCS multi-criteria score with **no hard gate and no freshness modelling** (every report is taken as current) |
| B5 | a language model reads the patient text and a table of the 12 nearest hospitals (reported capabilities, beds, data age) and picks one; no gate (subset of cases, real model calls) |
| CEDCS | freshness-discounted data, hard eligibility gate (unknown = provisional), priority-dependent ranking, 8 -> 25 km widening, re-verification, confidence |
| oracle | the nearest hospital that is truly appropriate (the evaluator sees the true state); a reference, not a policy |

For E1-E3, E5, E6 the decision core receives the **ground-truth triage**, so the decision layer is measured on its own. The language-model
stages (intake, triage, critic, advisor, narrator) are not part of these comparisons.

## Experiments

* **E1** all policies, three city loads, 2,000 cases each; plus B5 on a subset.
* **E2** freshness on / off with reported data aged 0, 1, 4, 12 hours.
* **E3** safety stress: with probability p (0 to 0.5) the triage handed to the policy is wrong (one non-ED REQUIRED capability dropped, priority
  lowered one level). Reported against the true profile, and as violations of the profile the policy was *given*.
* **E4** noise: a text degraded in four levels is triaged by the application's rule-based triage and routed; reports recall of the true
  REQUIRED set and appropriateness. The language-model triage is *not* evaluated here.
* **E4b** (not in the published results): the same degraded texts through the real AI triage, to compare with the rule-based triage. The command
  exists (`--llm-triage 40`, `--only e4_llm`) but the free Groq daily token limit was exhausted by the other AI experiments, so most calls failed;
  the numbers from that attempt (1 to 24 completed cases) are too few to report. Rerun it after the quota resets.
* **E5** weights: 20 random perturbations (each component x U(0.8, 1.2), renormalised) of Table 1; stability of the top choice and of undertriage.
* **E6** supply: each of six key capabilities (ICU bed, CT, cardiology, neurology, operating theatre, blood bank) is removed from each
  hospital with probability q (in the true state and the reports); reports radius widening, handoff and the safety property.
* **Latency** deterministic core per case, and the measured stage latencies of real full-pipeline runs saved by the application.

## Known limitations (state these in the paper)

1. Simulation, not deployment: no real EMS, hospital or outcome data. Absolute rates depend on the assumed world.
2. Hospital state changes are modelled as independent random walks; real occupancy has daily patterns, surges and correlated shocks.
3. Travel time is a distance estimate, treated as exact; real traffic adds error to every policy.
4. The requirement profile of each template is a plausible clinical mapping chosen by the authors; it was not validated by clinicians.
5. B4 is close to CEDCS by design (it uses the same score); the comparison isolates the hard gate and freshness, not the whole idea.
6. E1-E3, E5, E6 assume correct triage; E4 uses only the rule-based triage. How the language-model triage performs on real calls is not measured here.
7. The latency of the live pipeline depends on the external AI providers and their free-tier limits at the time of measurement.
8. Outcomes such as mortality or time to treatment are not measured: "appropriate" means capability coverage, a proxy for avoidable transfer.
