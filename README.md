# eeg_active_calibration

Neurotech@Berkeley, direction B (active calibration). For a new user, which images should they view during
calibration so a decoder pretrained on other subjects adapts best, counted in recorded presentations?

| File | What it is |
| --- | --- |
| `ubpkd.py` | UBP data, models, training and LOSO code from the UBP channel-distillation notebook (unchanged) |
| `activecal.py` | the experiment: offline acquisition simulator, adapters, policies, groups, summaries |
| `notebooks/activecal_colab.ipynb` | thin Colab runner: clone, data, Drive, one cell per group |
| `results/` | CSVs and plots committed back by the notebook's last cell |

## Protocol
* **Source decoder f0:** UBP EEGProject trained on every subject except the target (LOSO), uniform-11 blur targets.
* **Target:** simulated new user. 200 held-out training concepts are the evaluation set (never selectable); the other
  ~14,540 training images are the acquisition pool.
* **Evaluation:** 200-way retrieval on held-out concepts, mean of 10 draws (2,000 queries), with SE.
* **Acquisition:** `rep` = one presentation reveals one repetition; `image` = all 4 repetitions (4 presentations).
* **Rounds:** shared 25-image seed set, then ~2 rounds per doubling up to 3,200 presentations.
* **Policies:** random, category, kcenter, typiclust, error_region, expected_gain, intersubject, repeat_new.

## Groups (run in this order)
| Group | Subjects | What it decides |
| --- | --- | --- |
| `diag` | tuning subject 1 | data and method sanity: split, scale, identity check, full-pool ceilings incl. scratch |
| `gate` | tuning subject 1 | which adapter / acquisition beats f0, and at what budget (random policy only) |
| `main` | targets 2, 5, 9, 8 | policy comparison with the frozen adapter |

**Go:** a decision-based policy beats random and TypiClust on AUC for >= 3 of 4 targets, at several budgets, with the
noise check (repetition consistency) and the clean-gallery check passing. **Park:** coverage baselines tie it, the
winner prefers noisy trials, or random calibration never beats f0 (`gain_over_f0 <= 0`).

## Running
Open `notebooks/activecal_colab.ipynb` in Colab (File → Open notebook → GitHub), L4 GPU + High-RAM, add `GITHUB_TOKEN`
(fine-grained, this repo, Contents read/write) and `HF_TOKEN` as Colab secrets, run cells 1–9 in order.
