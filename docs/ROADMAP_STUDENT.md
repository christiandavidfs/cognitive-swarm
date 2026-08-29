# Roadmap — Student Iterativo por Ejecución

> Visión: modelo tiny sin datos, solo patrones. Conocimiento puede cambiar (números), razonamiento se acumula. Training por ejecución, iterativo, con guardrail.

## Estado actual (v8, 2026-08-29)
- **Truth 120/120 (100%)** determinístico: `executor 36/36 + string-op 12/12 + math 36/36 + reasoning 36/36 (20 patrones)` — `1.2s cold / 65ms warm`, `verified_memory.json` con `reasoning_op_sig`
- **Student v3 hybrid:** `TF-IDF word 1-2 + LogReg` sobre 226 samples (32 patrones ×5 sintéticos + 66 orig), `train 1.00 / CV 0.996 / routing 33/36 (91.7%) / novel 20/20 (100%) / 240KB / 0.26ms` — dispachador primario con fallback regex (`e847b6d` + `988379d`)
- **Branch:** `feat/student-v3-args` → `main` mergeado `v8-student-v3-20-20`, `paper.pdf 6p 205KB`

## Ciclo iterativo (cada 20 problemas nuevos)

```
1. Nuevo problema real → TruthRouter lo resuelve (hoy regex, mañana student)
   ↓ archiva en verified_memory con proveniencia {answer, tier, sources, disagreement}

2. Generador sintético → 5 paraphrases por patrón con números aleatorios (ej: 47 vs 100 handshake)
   Dataset = orig 66 + sintéticos → 226+

3. Re-entrena student (warm-start desde pkl) → TF-IDF + LogReg o Qwen2.5-0.5B LoRA
   Métricas: train_acc, CV 3-fold, routing 36, novel 20, size, lat_ms

4. Evalúa en 20 novel fijos (nunca vistos, números y wording distintos)
   Guardrail: si train 1.00 pero CV <0.9 o novel <90% → aborta, revisa distribución

5. Swap si novel ≥90% y routing ≥90% y CV ≥0.95 durante 2 ciclos → TruthRouter pasa a solo-student (sin fallback regex)
   Si no, mantiene hybrid y registra findings
```

## Milestones

| Milestone | Objetivo | Criterio de éxito | Docs a actualizar |
|---|---|---|---|
| **M0 (hoy)** | Student v3 hybrid 20/20 | `120/120 + 20/20` con fallback | `AGENTS.md`, `RESULTS.md` finding 5, `student_experiment.py` |
| **M1** | Student NER por patrón (v4 stub → aprendido) | `extract_args_learned` entrenado por patrón, `solo-student 15/20 → 20/20` sin heurística orden | `student_ner.py`, `RESULTS.md`, `ARCHITECTURE.md §6` |
| **M2** | Qwen2.5-0.5B LoRA para `pattern+args` seq2seq | `solo-student 120 + 20 novel ≥95%` sin regex, `<500ms`, `<100MB` | `THESIS.md`, `paper.tex`, `AGENTS.md` |
| **M3** | 20 problemas nuevos reales (no sintéticos) | Student generaliza a wording humano nuevo sin re-entrenar | `CONCLUSIONS.md`, `FINDINGS.md` |
| **M4** | Procedimiento memoria (no solo respuestas) | Cachea trazas de razonamiento verificadas (winning debate skeletons) | `ARCHITECTURE.md §10`, `memory/` |

## Cuándo usar student vs regex vs debate

- **Student (conf ≥0.07 y args válidos)** → usa student (ej: `47 conference handshake 0.15` donde regex falla)
- **Regex (fallback)** → si student `conf <0.07` o `args` incompletos → garantiza `36/36`
- **Debate (Tier 4)** → solo si `Tier 0-3` retornan `None` (problema sin patrón, abierto/contestado)

**Regla de promoción a solo-student:** `solo-student 120 + 20 novel ≥95%` durante 2 ciclos + `CV ≥0.95` + `size <500KB` + `lat <1ms`

## Registro de findings (actualizar en cada ciclo)

Cada ciclo registra en `docs/FINDINGS_STUDENT.md`: `dataset size, train/CV/routing/novel/size/lat, winner, ejemplos donde student supera regex, y donde falla`.

## Riesgos y mitigación

- **Olvido catastrófico:** siempre `combined = orig + sintéticos` (replay)
- **Colapso:** monitor `CV` y `novel` como guardrail, no solo `train`
- **8GB:** student tiny (<1MB) no compite con `Phi/Qwen` por RAM; carga en `ms`

---
*Última actualización: 2026-08-29 — v8 hybrid 20/20, próximo: M1 NER por patrón*
