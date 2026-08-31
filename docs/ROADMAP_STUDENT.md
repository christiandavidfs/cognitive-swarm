# Roadmap — Iterative Student by Execution

> Vision: tiny model with no data, only patterns. Knowledge can change (numbers), reasoning compounds. Training by execution, iterative, with guardrail.

## Current state (v9, 2026-08-30 — batch60 60/60)
- **Truth 120/120 (100%)** deterministic: `executor 36/36 + string-op 12/12 + math 36/36 + reasoning 36/36 (20 patterns)` — `1.2s cold / 65ms warm`, `verified_memory.json` with `reasoning_op_sig`. Nota: `truth_router_test.py --fresh` actual da 119/120 (1 fallo pre-existente, ver FINDINGS batch60); AGENTS dice 120/120 de benchmark anterior — desfase documentado, fuera de alcance de este fix.
- **Student v3 hybrid:** `TF-IDF word 1-2 + LogReg` on 226 samples (32 patterns ×5 synthetic + 66 orig), `train 1.00 / CV 0.996 / routing 33/36 (91.7%) / novel 20/20 (100%) / 240KB / 0.26ms` — primary dispatcher with regex fallback (`e847b6d` + `988379d`). Fix 2026-08-30: `stock_take_away` movido ANTES del student con `re.search` corregido → batch60 54→60/60 sin reentrenar, sin regresión 120/20novel.
- **Human batch:** `batch60_final.json 60/60 (100%)` condition A (deterministic regex fix); 100→200 human in progress. 15 stock breaking patterns (47 total) planned.
- **Branch:** `feat/student-v3-args` → `main` merged `v8-student-v3-20-20`, `paper.pdf 6p 205KB` (a recompilar v9 dual-router)

## Iterative cycle (every 20 new problems)

```
1. New real problem → TruthRouter resolves it (today regex, tomorrow student)
   ↓ archives in verified_memory with provenance {answer, tier, sources, disagreement}

2. Synthetic generator → 5 paraphrases per pattern with random numbers (e.g., 47 vs 100 handshake)
   Dataset = orig 66 + synthetic → 226+

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
| **M2 dual-router** | **TF-IDF vs Qwen2.5-0.5B LoRA** `pattern+args` seq2seq, **mismo dataset / mismas pruebas** | Ambos entrenados en **226+60 sintéticos** (32+1 patrones incl. `stock_take_away`); evaluados en **120 + 20 novel + 60 humano + 100 diverse** con mismo scorer; reportar `routing/novel/human/size/lat/CV` lado a lado para tesis | `qwen_lora_finetune.py`, `student_router.py`, `student_experiment.py --dual`, `THESIS.md`, `paper.tex`, `AGENTS.md` |
| **M3** | 100→200 human batch + stock 47 patterns | Student (TF-IDF y Qwen) generaliza a wording humano nunca visto sin reentrenar | `CONCLUSIONS.md`, `FINDINGS.md` |
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
