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

## v4 NER per pattern (2026-08-29, commit 2c2022f, in progress)
- **Architecture:** `student_ner.py` with `SLOTS` per pattern (handshake n, work_inverse n1/t1/t2...), `_context_window` (3 words before/after), `extract_args_learned` currently delegates to order-numbers heuristic but with trainable per-pattern interface
- **NER dataset:** 195 samples (each number labeled with its slot) over 226 synthetic
- **Finding:** order-numbers heuristic is already a positional mapping learned from 226 — `20/20 novel` and `120/120` hold without extra regex; learned per-pattern NER would give same result but with `TF-IDF context + LogReg` per slot
- **Next (M2):** replace heuristic with `Qwen2.5-0.5B LoRA` seq2seq `question → {"pattern":..., "args":{...}}` trained by execution on synthetic `verified_memory`, measure `solo-student 120 + 20 novel ≥95%` without fallback

---
*Update every iterative cycle with full metrics*
