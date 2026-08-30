# Dual-Router Benchmark — TF-IDF vs Qwen 0.5B

> **Objetivo tesis:** mantener ambos routers y medir sobre **las mismas pruebas** cuál generaliza mejor a wording nunca visto. No reemplazar TF-IDF — compararlo honestamente con Qwen.

## Por qué dual (tu idea es correcta)

- TF-IDF (240KB, 0.26ms, sklearn) ya es **proxy validado** del objetivo Qwen: mismo dataset (226), mismo objetivo (`question → pattern`), mismas métricas (CV, routing, novel). Barato, interpretable, corre en ms en 8GB sin contención.
- Qwen2.5-0.5B (≈600MB 4-bit MLX) aporta **semántica densa**: debería manejar paraphrases donde TF-IDF cae (ej. `same_weight` con "heavier/kilogram vs pound of X", `min_sisters` wording raro, `rate_machines` con verbos nuevos). El batch60 mostró que TF-IDF ya roba `take_away` → `simple_subtract`; Qwen debería distinguir mejor "take away 2 … how many do you have" vs "give away 4 … how many left".
- Mantener ambos permite en la tesis la tabla clave: *mismo entrenamiento, mismo scorer, ¿quién cae menos en 100 breaking?*

## Diseño — mismo dataset, mismas pruebas

### Dataset (igual para ambos)

- **Base:** `TRAINING_DATA` (66 orig, 32 patterns) + sintéticos 5/par patrón con números aleatorios.
- **v9+1:** añadir `stock_take_away` como clase 33 (2→5 samples sintéticos `"If you have {n} apples and you take away {k}, how many do you have?"` con n=5..10, k=2..5) y **rebalancear a ≥4 samples/clase** para evitar olvido catastrófico (FINDINGS: 29/32 clases con 2 samples → same_weight/min_sisters bajo umbral).
- Total: ~231 samples (33×5 + 66). Mismo `all_templates` en `student_experiment.py` (añadir `stock_take_away`).
- **Gold traces opcionales** (M2b): `student_trace.generate_trace` para Qwen `question → trace → answer` verificable; TF-IDF queda en `question → pattern`.

### Routers (misma interfaz)

```python
# Ambos exponen predict(question) -> (pattern, conf) y predict_with_args
TFIDFRouter:  TfidfVectorizer(1-2, 500 feats) + LogReg, pickle 240KB, THRESHOLD 0.07
QwenRouter:   Qwen2.5-0.5B-Instruct-4bit + LoRA r=8 alpha=16, <100MB LoRA, <500ms
              input:  "Question: ...\nPattern:"  output: '{"pattern": "...", "args": {...}}'
              fallback: si Qwen conf <0.07 o args inválidos → regex (mismo que TF-IDF)
```

### Harness de evaluación (mismo scorer)

`student_experiment.py --dual` (o `validate_dual.py`) corre **ambos** sobre:

| Suite | n | Qué mide |
|---|---|---|
| 120 closed | 120 | routing vs regex, answer exact via `resolve_*` |
| 20 novel (fixed) | 20 | números nunca vistos, wording distinto (handshake 47 etc.) |
| 60 human batch60 | 60 | wording humano real, incluye 6 take_away |
| 100 breaking | 100 | 15 stock fuera de 32 patrones — stress test |

Métricas lado a lado por router: `train_acc / CV 3-fold / routing 36 / novel 20/20 / human 60 / breaking 100 / size / lat_ms`. Guardar en `results_dual.json` para tesis.

### Entrenamiento

- **TF-IDF:** `student_router.train()` existente (sklearn, <1s).
- **Qwen LoRA (MLX):** `qwen_lora_finetune.py` real:
  - Base `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (~600MB, cabe en 8GB solo)
  - Datos: mismo 231 JSONL `{"prompt": "Question: ...\nPattern:", "completion": '{"pattern": "handshake", "args": {"n": 47}}'}`
  - LoRA r=8, lr 2e-4, 3 epochs, batch 4, dropout 0.05 — sobre MLX `mlx_lm.lora`
  - Verificación: cada trace/step pasa `verify_trace` / executor antes de contar como correcto
  - Si 8GB no da para entrenar local, entrenar en proxy TF-IDF y dejar Qwen como evaluación con prompt (few-shot) mientras tanto — honest limits.

## Plan de implementación (iterativo, sin romper 8GB)

1. **Ahora (sin Qwen weights):** añadir `stock_take_away` a `TRAINING_DATA` + `all_templates`, reentrenar TF-IDF rebalanceado a ≥4/clase, medir `60/60` y `20 novel` sin regresión. Esto valida condición B (student aprende take_away).
2. **Scaffold dual:** extender `student_experiment.py` con `--dual` que, si no hay `qwen_router.pkl`, reporta `Qwen: not trained (proxy)` pero ya genera la tabla comparativa (TF-IDF real vs placeholder).
3. **Qwen real:** implementar `tools/qwen_router.py` (MLX load, LoRA, predict con misma firma) y `qwen_lora_finetune.py` real siguiendo `student_trace` gold traces. Entrenar y medir `<500ms`.
4. **Tesis:** tabla dual en RESULTS.md + figura `dual_router.png` (TF-IDF vs Qwen en 120/20/60/100).

## Qué esperamos honestamente

- TF-IDF seguirá ganando en **lat/size** (240KB vs ~100MB, 0.26ms vs ~300ms) y en **CV** si dataset pequeño.
- Qwen debería ganar en **100 breaking** y en paraphrases tipo "heavier/kilogram" donde TF-IDF cae a `fraction_compare`.
- Si Qwen no gana, es hallazgo honesto: TF-IDF ya aprende estructura para este dominio — reportarlo así.

## Siguiente paso inmediato

¿Apruebo el paso 1+2 ahora (rebalancear TRAINING_DATA + scaffold dual sin descargar Qwen)? Queda listo para que luego entrenemos Qwen cuando haya ventana de tiempo/RAM.
