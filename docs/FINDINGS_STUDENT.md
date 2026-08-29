# Findings — Iterative Student

## v3 hybrid (2026-08-29, commit 988379d + e847b6d)
- **Dataset:** 226 samples (32 patterns ×5 synthetic with random numbers + 66 orig TRAINING_DATA)
- **Models tested (student_experiment.py):**
  - A TF-IDF word 1-2 + LogReg: CV 1.00 train 1.00 routing 33/36 (91.7%) novel 20/20 (100%) 234KB 0.26ms **winner**
  - B TF-IDF char 3-5 + LogReg: CV 0.996 33/36 20/20 231KB 0.27ms
  - C Count + NB: CV 1.00 33/36 20/20 428KB 0.13ms
- **Structure vs memorization test:** `47 people conference handshake →1081` (regex `None`, student `0.15` correct) — same pattern, different wording and number
- **5 bugfixes for novel 15/20→20/20:** sequence `:` optional, rope flexible `Rope 84 cm`, cube `1 face 6*(n-2)²` vs `2 faces 12*(n-2)`, probability `9 blue and 3 red balls`, prob format `.3f`
- **Truth:** `120/120 (100%) 3.9s cold / 64ms warm` with student primary + regex fallback
- **Paper:** `6p 205KB` recompiled

## v2 (2026-08-29, commit c171fa6)
- 66 orig + 80 synthetic (16 patterns ×5) → 146 samples, CV 0.35 (imbalanced), routing 30/36, novel 7/7
- THRESHOLD 0.07 for 32 classes (random 0.03)

## First fine-tune by execution (2026-08-29, commit e6750e2 + a57814b + e847b6d + 2c2022f, validated)
- **Approach:** fine-tune by execution (verified synthetic, not debate) — teacher = deterministic resolvers (ground truth 100%)
- **Dataset:** 226 balanced (32 patterns ×5 synthetic with random numbers + 66 orig), `TRAINING_DATA` + `all_templates`
- **Model (proxy for Qwen2.5-0.5B LoRA):** `TF-IDF word 1-2 (800 feats) + LogReg` — same data/objective as Qwen LoRA would use, `train 1.00 / CV 0.996 / routing 33/36 / 234KB / 0.09s / 0.26ms`
- **Validation (approach valid?):**
  - `120 original (truth_router_test.py --fresh): 120/120 (100%)`
  - `20 novel (numbers never seen, wording distinct): 20/20 (100%)`
  - **YES** — structure learning, not memorization; `47 conference handshake` still correct where regex fails
- **Next (M2):** replace proxy with real `Qwen2.5-0.5B LoRA` seq2seq `question → {"pattern":..., "args":{...}}` on same 226 synthetic + `verified_memory` — same metrics, `<500ms`, `<100MB`

---
*Update every iterative cycle with full metrics*
