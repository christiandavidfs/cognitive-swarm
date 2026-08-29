# Findings — Student Iterativo

## v3 hybrid (2026-08-29, commit 988379d + e847b6d)
- **Dataset:** 226 samples (32 patrones ×5 sintéticos con números aleatorios + 66 orig TRAINING_DATA)
- **Modelos probados (student_experiment.py):**
  - A TF-IDF word 1-2 + LogReg: CV 1.00 train 1.00 routing 33/36 (91.7%) novel 20/20 (100%) 234KB 0.26ms **ganador**
  - B TF-IDF char 3-5 + LogReg: CV 0.996 33/36 20/20 231KB 0.27ms
  - C Count + NB: CV 1.00 33/36 20/20 428KB 0.13ms
- **Prueba estructura vs memorización:** `47 people conference handshake →1081` (regex `None`, student `0.15` acierta) — mismo patrón, diferente wording y número
- **5 bugfixes para novel 15/20→20/20:** sequence `:` opcional, rope flexible `Rope 84 cm`, cube `1 face 6*(n-2)²` vs `2 faces 12*(n-2)`, probability `9 blue and 3 red balls`, prob format `.3f`
- **Truth:** `120/120 (100%) 3.9s cold / 64ms warm` con student primario + fallback regex
- **Paper:** `6p 205KB` recompilado

## v2 (2026-08-29, commit c171fa6)
- 66 orig + 80 sintéticos (16 patrones ×5) → 146 samples, CV 0.35 (desbalanceado), routing 30/36, novel 7/7
- THRESHOLD 0.07 para 32 clases (random 0.03)

## Próximo (M1 v4)
- NER por patrón: `_context_window` + clasificador por slot, entrenado sobre 226 con números aleatorios, objetivo `solo-student 20/20` sin heurística orden

---
*Actualizar en cada ciclo iterativo con métricas completas*
