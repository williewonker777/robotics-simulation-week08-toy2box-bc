# E6 post-hoc diagnosis (read-only)

> Exploratory, unpaired analysis. It does not alter the frozen H4 verdict or inputs.

## Outcome and first failed milestone

- E2: 40/200 successes; first near close 167, first secured 78, first failed 89, never near 33.
- E6: 27/200 successes; first near close 176, first secured 48, first failed 128, never near 24.
- First failed milestone: **grasp acquisition**. The policy closes near the truck but does not reach the existing 40-step secured criterion.

## First near-close metrics (all near-close trials)

| metric | E2 median | E6 median | E6−E2 | exploratory p |
|---|---:|---:|---:|---:|
| step | 312.000 | 294.000 | -18.000 | 8.71e-05 |
| height | 13.631 mm | 14.351 mm | +0.721 mm | 0.421 |
| XY offset | 12.739 mm | 18.412 mm | +5.673 mm | 3.76e-05 |
| |heading error| | 2.982° | 8.158° | +5.177° | 8.57e-10 |
| |gripper pos| | 0.510 | 0.514 | +0.003 | 3.83e-05 |

The primary gap uses every first-near-close trial, without selecting successes or secured grasps. E6 has **more**, not fewer, near-close opportunities. The heading gap is positive in all four initial-|yaw| strata and remains present separately in secured and failed cohorts. Initial batches are unpaired; no trial-paired claim is made.

The first near-close action occurs earlier in E6 (median step 294 versus 312). The exploratory physical-gripper threshold latency is unchanged at a median of one step, so the timing gap is the policy's earlier close decision rather than slower gripper response. Among the postselected secured cohorts, acquisition-run start delay is 18.0 versus 25.5 steps; this is reported descriptively and is not used to explain the loss because failed grasps never enter that cohort.

## Stage distribution

| condition | success | placement | transport | grasp | approach |
|---|---:|---:|---:|---:|---:|
| E2 | 40 | 5 | 53 | 75 | 27 |
| E6 | 27 | 2 | 31 | 120 | 20 |

## Interpretation

- The -5 mm intervention transferred to generated height, but **not** to policy close height: E6 rollout height is slightly higher than E2.
- Generated first-close XY/heading precision is nearly unchanged (E2 3.705 mm/0.714°, E6 3.337 mm/0.717°), so the rollout alignment loss is not inherited from imprecise generated close poses.
- E6 closes near the truck more often, but first acquisition drops sharply. The concrete new gap is worse wrist heading and lateral placement at that first close, before transport.
- This is not explained by filtering only successful/secured episodes: the same directional heading gap appears in all near-close trials, all four yaw strata, and both secured and failed outcome strata.

## At-most-one H5 recommendation

- **Single factor:** fixed open-gripper alignment dwell immediately before first close.
- Intervention: start from stock E2 natural60 generation (no Z offset) and repeat the final already-aligned, open-gripper waypoint for one preregistered fixed duration. Keep pose geometry, source selection, close/lift/place segments, data count, BC, and evaluation unchanged.
- Rationale: E6 closes 18 steps earlier and reaches the truck with worse heading/XY error even though generated close poses remain precise. Repeated aligned pre-close samples target attainment of that state rather than changing geometry again.
- Gate: freeze one duration; require generated first-close XY/heading/height to stay within a narrow preregistered E2 tolerance and exact dwell count. Require later close timing, lower all-near heading/XY error, and higher first-secured fraction before interpreting performance.
- Caution: this is post-hoc. E1 showed heading improvement alone need not raise success; reject H5 unless acquisition and performance improve under frozen criteria.

Exact distributions, exploratory tests, hashes, and per-stage counts are in `posthoc_e6_diagnosis.json`.
