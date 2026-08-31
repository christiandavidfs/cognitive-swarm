# Roadmap — Iterative Student by Execution

> Vision: tiny model with no data, only patterns. Knowledge can change (numbers), reasoning compounds. Training by execution, iterative, with guardrail.

## Current state (v9, 2026-08-30 — batch60 60/60)
- **Truth 120/120 (100%)** deterministic: `executor 36/36 + string-op 12/12 + math 36/36 + reasoning 36/36 (20 patterns)` — `1.2s cold / 65ms warm`, `verified_memory.json` with `reasoning_op_sig`. Nota: `truth_router_test.py --fresh` actual da 119/120 (1 fallo pre-existente, ver FINDINGS batch60); AGENTS dice 120/120 de benchmark anterior — desfase documentado, fuera de alcance de este fix.
- **Student v4 hybrid:** `TF-IDF word 1-2 + LogReg 800 feats` on **132 TRAINING (33×4) +165 synthetic =297**, `train 1.00 / CV 0.962 / routing 31-32/36 / novel 20/20 (100%) / 247KB / 0.26ms` — primary dispatcher with hybrid (`stock_take_away` before student, regex fallback). Fix 2026-08-30: `stock_take_away` before student + rebalanced 33×4 (was 32×2) → batch60 54→60/60, e-r-11 disambiguated, no regression.
- **Human batch:** `batch60_final.json 60/60 (100%)` condition A (deterministic regex fix); 100→200 human in progress. 15 stock breaking patterns (47 total) planned.
- **Branch:** `feat/student-v3-args` → `main` merged `v8-student-v3-20-20`, `paper.pdf 6p 205KB` (a recompilar v9 dual-router)

## Iterative cycle (every 20 new problems)

```
1. New real problem → TruthRouter resolves it (today regex, tomorrow student)
   ↓ archives in verified_memory with provenance {answer, tier, sources, disagreement}

2. Synthetic generator → 5 paraphrases per pattern with random numbers (e.g., 47 vs 100 handshake)
   Dataset = orig 132 (33×4) + synthetic 165 (33×5) → 297

3. Re-train student (warm-start from pkl) → TF-IDF + LogReg or Qwen2.5-0.5B LoRA
   Metrics: train_acc, CV 3-fold, routing 36, novel 20, size, lat_ms

4. Evaluate on 20 fixed novel (never seen, different numbers and wording)
   Guardrail: if train 1.00 but CV <0.9 or novel <90% → abort, check distribution

5. Swap if novel ≥90% and routing ≥90% and CV ≥0.95 for 2 cycles → TruthRouter switches to student-only (no regex fallback)
   Otherwise keeps hybrid and logs findings
```

## Milestones

| Milestone | Goal | Success criterion | Docs to update |
|---|---|---|---|
| **M0 (today)** | Student v3 hybrid 20/20 | `120/120 + 20/20` with fallback | `AGENTS.md`, `RESULTS.md` finding 5, `student_experiment.py` |
| **M1** | Student NER per pattern (v4 stub → learned) | `extract_args_learned` trained per pattern, `solo-student 15/20 → 20/20` without order heuristic | `student_ner.py`, `RESULTS.md`, `ARCHITECTURE.md §6` |
| **M2 dual-router** | **TF-IDF vs Qwen2.5-0.5B LoRA** `pattern+args` seq2seq, **same dataset / same tests** | Both trained on **132 TRAINING +165 synthetic =297 (33 patterns incl. `stock_take_away`)**; evaluated on **120 + 20 novel + 60 human + 100 diverse** with same scorer; report `routing/novel/human/size/lat/CV` side-by-side for thesis | `qwen_lora_finetune.py`, `student_router.py`, `student_experiment.py --dual`, `THESIS.md`, `paper.tex`, `AGENTS.md` |
| **M3** | 100→200 human batch + stock 47 patterns | Student (TF-IDF and Qwen) generalizes to never-seen human wording without retraining | `CONCLUSIONS.md`, `FINDINGS.md` |
| **M4** | Procedure memory (not just answers) | Cache verified reasoning traces (winning debate skeletons) | `ARCHITECTURE.md §10`, `memory/` |

## When to use student vs regex vs debate

- **Student (conf ≥0.07 and valid args)** → use student (e.g., `47 conference handshake 0.15` where regex fails)
- **Regex (fallback)** → if student `conf <0.07` or incomplete `args` → guarantees `36/36`
- **Debate (Tier 4)** → only if `Tier 0-3` return `None` (no pattern, open/contested problem)

**Promotion rule to student-only:** `solo-student 120 + 20 novel ≥95%` for 2 cycles + `CV ≥0.95` + `size <500KB` + `lat <1ms`

## Findings log (update every cycle)

Each cycle logs in `docs/FINDINGS_STUDENT.md`: `dataset size, train/CV/routing/novel/size/lat, winner, examples where student beats regex, and where it fails`.

## Risks and mitigation

- **Catastrophic forgetting:** always `combined = orig + synthetic` (replay)
- **Collapse:** monitor `CV` and `novel` as guardrail, not just `train`
- **8GB:** tiny student (<1MB) does not compete with `Phi/Qwen` for RAM; loads in `ms`

---
*Last updated: 2026-08-29 — v8 hybrid 20/20, next: M1 NER per pattern*
