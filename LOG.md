# EEG Active Calibration (Direction B): Experiment Log

Neurotech@Berkeley, Project B. Tuning on subject 1, policy comparison on subjects 2, 5, 8, 9. Updated 2026-11-06.
Code: `github.com/navyers21/eeg_active_calibration` (branch `main`, `activecal.py` v0.2 for main, v0.3 for the step check, v0.4 for the oracle).

## Context

|  |  |
| --- | --- |
| **Problem** |  A new user starts with a decoder pretrained on other subjects. Which images should they view during calibration so the decoder adapts best? If a smart choice of images beats a random one, calibration can be shorter. |
| **Dataset** | THINGS-EEG, 10 subjects. Train: 16,540 images (1,654 concepts × 10), 4 repetitions each. The official test set is not used. |
| **Task** | Zero-shot retrieval, 200-way, in CLIP (RN50) space with UBP uniform-11 blur targets. **Chance = 0.5%.** |
| **Source decoder (f0)** | UBP EEGProject trained on the 9 other subjects (LOSO), never on the target. This is the uncalibrated starting point. |
| **Evaluation set** | The target's 200 held-out training concepts (all 10 images each), never selectable. Each score is the mean of 10 draws (one image per concept per draw, 2,000 queries), 4-rep averages. |
| **Acquisition pool** | The target's other ~14,540 training images. A policy picks images; the simulator reveals that user's real recorded EEG for them. |
| **Budget unit** | Presentations. One image = 4 presentations (all 4 repetitions revealed at once). Budgets 100 to 3,200 (25 to 800 images). Every run starts from the same 25 random images per seed. |
| **Seeds** | 3 seeds per target (main), 2 seeds (gate). Numbers are seed means. |

## Simulation

No new EEG is recorded. Each training subject already viewed every pool image 4 times, so a "calibration session" is replayed from those recordings: when a policy asks to show an image, the simulator hands back that subject's real EEG for it. The policy never sees EEG for images it has not asked for.

| Step | What happens |
| --- | --- |
| **1. Pick a target** | One subject plays the new user (e.g. subject 2). f0 is the UBP decoder trained on the other 9 subjects. |
| **2. Split the target's data** | 200 concepts (all their images) are locked away as the evaluation set. The remaining ~14,540 images form the pool the policy can choose from. |
| **3. Seed round** | 25 random pool images are "shown" (same 25 for every policy on a given seed): 100 presentations. |
| **4. Adapt** | Starting from a fresh copy of f0, the adapter trains on every image acquired so far (input = mean of its 4 repetitions, target = its blurred CLIP feature). Each round restarts from f0, it does not continue the last round's model. |
| **5. Score** | If the total has hit a budget (100, 200, ..., 3,200), the adapted model is evaluated on the locked 200 concepts and a row is logged. |
| **6. Choose next images** | The policy looks at what it is allowed to see (image features, the target's EEG for already-acquired images, the other subjects' recordings, the current adapted model) and picks the next batch from unseen pool images. Batch size is about half the current total, so there are ~2 rounds per doubling (100 → 150 → 200 → 300 → 400 ...). |
| **7. Reveal** | The chosen images' 4 recorded repetitions are added to the acquired set (+4 presentations each). Back to step 4 until 3,200. |

One run = one target × montage × policy × adapter × seed and gives a learning curve of 6 points (top-1 at 100 to 3,200). Every policy is scored on the same evaluation set, so curves can be compared directly. f0 alone (0 presentations) and the full pool (every pool image, 1,000 steps) bracket each curve: the first is the floor calibration has to beat, the second is the most calibration could ever give.

**Groups**

| Group | Subjects | Runs | Decides |
| --- | --- | --- | --- |
| diag | 1 | f0 + 3 full-pool ceilings per montage | Is the pipeline sane, and what is the ceiling? |
| gate | 1 | random policy × 11 adapter settings × 2 seeds | Which adapter to freeze, and does calibration beat f0 at all? |
| main | 2, 5, 8, 9 | 8 policies × 3 seeds, frozen adapter | Does any policy beat random? |
| oracle | 1 | random vs 2 oracle policies × 2 seeds, frozen adapter | Is there any headroom for image selection at all? |

## Legend

**Montages**

| Name | Ch | Channels |
| --- | --- | --- |
| occipital8 | 8 | O1, Oz, O2, PO7, PO3, POz, PO4, PO8 |
| posterior17 | 17 | UBP / NeuroBridge posterior set (P/PO/O rows) |

**Policies** (how the next images are chosen)

| Name | Type | Rule |
| --- | --- | --- |
| random | baseline | Uniform over unseen pool images |
| category | coverage | One image from each least-covered concept |
| kcenter | coverage | Farthest image from everything acquired, in CLIP space |
| typiclust | coverage | Most typical image of each uncovered CLIP cluster (strong published baseline) |
| error_region | decision | Images near acquired ones the current decoder fits worst |
| expected_gain | decision | Largest first-layer gradient, using the other subjects' mean EEG as a stand-in response |
| intersubject | decision | Images where the other subjects' f0 embeddings disagree most |
| repeat_new | decision | Half re-show worst-fit images, half kcenter. With whole-image acquisition this reduces to kcenter |
| oracle_loss | oracle (cheats) | Unseen images the current decoder fits worst, scored with their real EEG before they are shown |
| oracle_batch | oracle (cheats) | Draws 8 random candidate batches, adapts on each with its real EEG, keeps the batch that most improves retrieval on 500 pool images set aside for the oracle (never the evaluation concepts) |

**Adapters** (how f0 is updated on the acquired data)

| Name | Meaning |
| --- | --- |
| lowrank-r8 / r32 | Only a rank-8 or rank-32 update to the first linear layer |
| full-a10 | First linear layer trained, L2 pull back to f0 (weight 10) |
| all-a1 | Every layer trained, L2 pull back to f0 (weight 1) |
| reh0.5 | Half of each batch is rehearsal: the other subjects' EEG on pool images |
| scratch | Fresh model, target data only (does the f0 prior help at all?) |
| s25 / s50 / s100 / s200 | Training steps per calibration round (batch 64) |
| sel | Step count picked from {0, 25, 50, 100, 200} on 20% of the acquired images held out; 0 = keep f0 |
| rep / image | rep = one presentation reveals one repetition; image = all 4 |

**Columns:** top-1 = retrieval accuracy (%). **AUC** = mean top-1 over log2(presentations), 100 to 3,200. **Gain vs f0** = AUC minus f0 top-1 (must be > 0 for calibration to be worth doing). **vs random** = paired AUC difference against random on the same target (seeds averaged first). **Wins** = targets with a positive difference, out of 4. Seed spread is about ±0.4 pts, so differences under about 0.5 pts are not established.

## Results: diagnostics (subject 1)

| Montage | f0 | Full pool, frozen adapter | Full pool, all layers | Full pool, scratch |
| --- | --- | --- | --- | --- |
| occipital8 | 4.70 | 14.95 | 14.95 | 14.30 |
| posterior17 | 5.65 | 15.75 | 17.20 | 16.25 |

Full pool = all ~14,540 pool images, 1,000 steps. Identity check (adapter at 0 steps = f0) passed. With all the data, scratch nearly matches adapting f0, so the adapter is not the bottleneck.

## Results: gate (subject 1, random policy)

| Group | Montage | Adapter | 100 | 200 | 400 | 800 | 1600 | 3200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f0 | occipital8 |  | 4.70 | 4.70 | 4.70 | 4.70 | 4.70 | 4.70 |
| Adapters | occipital8 | lowrank-r8, rep | 2.05 | 1.90 | 2.98 | 3.28 | 4.40 | 4.62 |
|  | occipital8 | lowrank-r8 | 2.58 | 2.85 | 2.88 | 2.80 | 4.65 | 5.68 |
|  | occipital8 | lowrank-r32 | 2.15 | 2.25 | 3.20 | 3.25 | 4.32 | 5.25 |
|  | occipital8 | full-a10 | 3.75 | 4.95 | 4.68 | 4.88 | 5.82 | 6.22 |
|  | occipital8 | all-a1 | 2.88 | 3.62 | 4.22 | 4.70 | 5.82 | 6.35 |
|  | occipital8 | **all-a1, reh0.5 (frozen)** | 4.38 | 4.92 | 4.65 | 5.35 | 5.65 | 6.30 |
|  | occipital8 | scratch | 0.88 | 0.90 | 1.12 | 1.67 | 2.30 | 3.35 |
| Steps | occipital8 | all-a1, reh0.5, s25 | 3.98 | 4.80 | 4.65 | 5.10 | 5.32 | 5.95 |
|  | occipital8 | all-a1, reh0.5, s50 | 4.70 | 4.72 | 4.85 | 4.78 | 5.58 | 6.60 |
|  | occipital8 | all-a1, reh0.5, s100 | 4.80 | 4.55 | 4.68 | 4.68 | 5.90 | 6.52 |
|  | occipital8 | all-a1, reh0.5, sel | 4.20 | 4.68 | 4.62 | 5.40 | 5.18 | 6.68 |
| f0 | posterior17 |  | 5.65 | 5.65 | 5.65 | 5.65 | 5.65 | 5.65 |
| Adapters | posterior17 | lowrank-r8, rep | 1.78 | 2.68 | 2.97 | 3.22 | 4.32 | 5.10 |
|  | posterior17 | lowrank-r8 | 2.55 | 2.65 | 2.80 | 3.00 | 3.98 | 5.85 |
|  | posterior17 | lowrank-r32 | 2.08 | 2.25 | 3.20 | 3.30 | 3.85 | 5.93 |
|  | posterior17 | full-a10 | 3.35 | 4.28 | 4.60 | 5.22 | 6.18 | **7.65** |
|  | posterior17 | all-a1 | 2.80 | 2.68 | 3.80 | 4.78 | 5.38 | 7.32 |
|  | posterior17 | **all-a1, reh0.5 (frozen)** | 4.38 | 4.68 | 4.95 | 5.28 | 5.88 | 6.80 |
|  | posterior17 | scratch | 0.57 | 0.75 | 0.88 | 0.92 | 2.10 | 3.60 |
| Steps | posterior17 | all-a1, reh0.5, s25 | 4.18 | 5.25 | 4.93 | 5.72 | 5.80 | 5.88 |
|  | posterior17 | all-a1, reh0.5, s50 | 4.72 | 4.55 | 4.53 | 5.18 | 6.32 | 6.48 |
|  | posterior17 | all-a1, reh0.5, s100 | 4.38 | 4.85 | 5.05 | 5.30 | 5.75 | 7.28 |
|  | posterior17 | all-a1, reh0.5, sel | 4.62 | 5.18 | 4.03 | 5.28 | 6.12 | 7.32 |

The frozen adapter (all-a1, reh0.5, s200) was chosen on this table and used for the main group. The step rows were run afterwards as a check.

## Results: main (subjects 2, 5, 8, 9; frozen adapter)

**f0 and full-pool ceilings per target**

| Target | Montage | f0 | Full pool | Full pool, all layers | Full pool, scratch |
| --- | --- | --- | --- | --- | --- |
| 2 | occipital8 | 6.60 | 13.45 | 13.30 | 12.65 |
| 2 | posterior17 | 7.05 | 12.85 | 12.65 | 11.55 |
| 5 | occipital8 | 4.20 | 8.50 | 8.45 | 7.70 |
| 5 | posterior17 | 4.90 | 9.65 | 9.60 | 9.55 |
| 8 | occipital8 | 5.30 | 18.50 | 18.70 | 18.00 |
| 8 | posterior17 | 7.30 | 24.05 | 23.95 | 22.75 |
| 9 | occipital8 | 5.35 | 11.95 | 11.90 | 11.50 |
| 9 | posterior17 | 5.65 | 13.05 | 12.75 | 11.75 |

**Policy comparison**

| Group | Montage | Policy | AUC | Gain vs f0 | vs random | Wins | vs typiclust | Wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | occipital8 | random | 5.32 | −0.04 |  |  | +0.12 | 3/4 |
| Coverage | occipital8 | category | 5.28 | −0.09 | −0.04 | 2/4 | +0.08 | 3/4 |
|  | occipital8 | typiclust | 5.20 | −0.16 | −0.12 | 1/4 |  |  |
|  | occipital8 | kcenter | 4.71 | −0.66 | −0.61 | 0/4 | −0.49 | 0/4 |
| Decision | occipital8 | intersubject | 5.05 | −0.31 | −0.27 | 0/4 | −0.15 | 0/4 |
|  | occipital8 | error_region | 4.99 | −0.37 | −0.33 | 0/4 | −0.21 | 0/4 |
|  | occipital8 | repeat_new | 4.80 | −0.56 | −0.51 | 0/4 | −0.39 | 0/4 |
|  | occipital8 | expected_gain | 4.75 | −0.61 | −0.57 | 0/4 | −0.45 | 0/4 |
| Baseline | posterior17 | random | 5.31 | −0.91 |  |  | +0.12 | 4/4 |
| Coverage | posterior17 | category | 5.21 | −1.01 | −0.10 | 1/4 | +0.02 | 3/4 |
|  | posterior17 | typiclust | 5.20 | −1.03 | −0.12 | 0/4 |  |  |
|  | posterior17 | kcenter | 4.89 | −1.34 | −0.43 | 0/4 | −0.31 | 0/4 |
| Decision | posterior17 | error_region | 5.13 | −1.10 | −0.19 | 1/4 | −0.07 | 2/4 |
|  | posterior17 | intersubject | 4.96 | −1.27 | −0.35 | 0/4 | −0.24 | 1/4 |
|  | posterior17 | repeat_new | 4.84 | −1.39 | −0.47 | 0/4 | −0.36 | 0/4 |
|  | posterior17 | expected_gain | 4.83 | −1.40 | −0.49 | 0/4 | −0.37 | 0/4 |

**Top-1 by budget** (mean of 4 targets × 3 seeds)

| Montage | Policy | 100 | 200 | 400 | 800 | 1600 | 3200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| occipital8 | random | 4.38 | 4.82 | 4.85 | 5.20 | 6.04 | 7.00 |
| occipital8 | typiclust | 4.52 | 4.68 | 4.65 | 5.23 | 5.73 | 6.89 |
| occipital8 | category | 4.40 | 4.74 | 4.89 | 5.22 | 5.85 | 6.94 |
| occipital8 | error_region | 4.53 | 4.62 | 4.52 | 4.87 | 5.47 | 6.41 |
| occipital8 | expected_gain | 4.44 | 4.58 | 4.58 | 4.58 | 4.88 | 5.82 |
| occipital8 | kcenter | 4.43 | 4.31 | 4.42 | 4.58 | 5.07 | 5.87 |
| posterior17 | random | 4.51 | 4.58 | 4.87 | 5.33 | 6.02 | 7.02 |
| posterior17 | typiclust | 4.43 | 4.54 | 4.74 | 5.27 | 5.68 | 7.08 |
| posterior17 | category | 4.43 | 4.72 | 4.93 | 5.12 | 5.63 | 6.90 |
| posterior17 | error_region | 4.49 | 4.68 | 4.84 | 5.02 | 5.58 | 6.55 |
| posterior17 | expected_gain | 4.48 | 4.50 | 4.50 | 4.96 | 5.02 | 5.82 |
| posterior17 | kcenter | 4.33 | 4.63 | 4.62 | 4.70 | 5.19 | 6.26 |

**Oracle upper bound (subject 1, 2 seeds).** The oracle policies are allowed to look at a candidate image's real EEG before choosing it, which no real calibration session could do. They estimate how much any selection rule could gain over random. If even they do not beat random, image selection has no headroom on this data.

| Montage (f0) | Policy | 100 | 200 | 400 | 800 | 1600 | 3200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| occipital8 (4.70) | random | 4.57 | 4.90 | 4.88 | 5.22 | 5.85 | 6.75 |
|  | oracle_batch | 4.55 | 4.65 | 5.20 | 5.22 | 5.50 | 6.78 |
|  | oracle_loss | 4.35 | 4.22 | 4.62 | 5.22 | 5.05 | 6.18 |
| posterior17 (5.65) | random | 4.78 | 4.70 | 5.35 | 5.55 | 5.90 | 6.25 |
|  | oracle_batch | 4.15 | 4.68 | 4.72 | 5.25 | 5.82 | 7.10 |
|  | oracle_loss | 5.03 | 4.55 | 4.90 | 4.70 | 6.38 | 7.15 |

Neither oracle beats random below 3,200 presentations on either montage. Both oracles are about 0.9 pts above random at 3,200 on posterior17 only, which needs more seeds to establish.

**Random gain over f0 by budget:** occipital8 −0.98 / −0.55 / −0.52 / −0.16 / +0.67 / +1.63; posterior17 −1.72 / −1.65 / −1.35 / −0.90 / −0.20 / +0.79 (100 to 3,200 presentations).

**Noise check** (mean repetition consistency of acquired images; pool mean about 0.074 occipital8, 0.043 posterior17): kcenter and repeat_new 0.080 / 0.048, all others 0.072 to 0.076 / 0.040 to 0.044. Clean-gallery top-1 ranks policies the same way (random 4.34 / 4.12 highest, expected_gain 3.85 / 3.81 lowest).

## Takeaways

1. **No policy beats random.** Every decision-based policy loses to random on AUC (0/4 or 1/4 targets), and the gap grows with budget (about −1 pt at 3,200 for expected_gain and kcenter). Coverage baselines tie random at best. Even the oracle policies, which see each candidate's real EEG, do not beat random below 3,200 presentations, so the problem is a lack of headroom, not weak policies.
2. **Calibration does not beat f0 until about 1,600 presentations.** Random's gain over f0 is negative at every budget up to 800 (≈ 200 images, ~10 min) on both montages.
3. **Fewer steps removes the harm but adds no gain.** At 50 to 100 steps the adapter matches f0 at 100 presentations instead of dropping below it, but nothing clearly beats f0 before 1,600.
4. **Held-out step selection fails at small budgets.** With 5 to 20 held-out images it picks harmful step counts (4.03 vs f0 5.65), even though keeping f0 was an option. A real session could not tell when to stop either.
5. **The f0 prior helps at small budgets, not at large ones.** Scratch is far below f0 under 800 presentations (0.9% at 100) but matches adapting f0 on the full pool.
6. **Go/no-go: park.** The precondition (random calibration beats f0) fails at practical budgets, and no policy beats random or TypiClust on ≥3 of 4 targets. The oracle shows there is little to gain from selection even in principle. Calibration time is better spent on more trials than on choosing them.

**Caveats:** evaluation uses held-out training concepts with 4-rep averages, so absolute numbers are not comparable to test-set numbers (80-rep averages); f0 was early-stopped on the source subjects' copies of the same 200 held-out concepts, so it is slightly optimistic; source data use `hf_whiten` (63-channel whitening leaks across channels); 4 targets, 3 seeds; gate and step check on one subject with 2 seeds; repeat_new is not a real re-showing test under whole-image acquisition; rehearsal sampling is not seeded; oracle_batch compares only 8 candidate batches per round on a 4-rep check set, so it is a weak oracle, and the oracle was run on one subject with 2 seeds.