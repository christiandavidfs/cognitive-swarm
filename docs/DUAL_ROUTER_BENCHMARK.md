# Dual-Router Benchmark — TF-IDF vs Qwen 0.5B

> **Thesis goal:** keep both routers and measure on the **same tests** which generalizes better to unseen wording. Do not replace TF-IDF — compare honestly vs Qwen. This is the apples-to-apples table for the thesis: same training, same scorer, who drops less on 100 breaking?

## Why dual

- **TF-IDF (240KB, 0.26ms, sklearn)** is a **validated proxy** for the Qwen objective: same dataset, same objective (`question → pattern`), same metrics (CV, routing, novel). Cheap, interpretable, runs in ms on 8GB without contention.
- **Qwen2.5-0.5B (≈600MB 4-bit MLX)** brings dense semantics: should handle paraphrases where TF-IDF drops (e.g. `same_weight` with "heavier/kilogram vs pound of X", `min_sisters` wording variants, `rate_machines` with new verbs). Batch60 showed TF-IDF confused `take_away` → `simple_subtract`; Qwen should better distinguish "take away 2 … how many do you have" vs "give away 4 … how many left".
- Keeping both yields the key thesis table: same training, same scorer, dual comparison vs phi / qwen-3B / minimax-m3 frontier.

## Design — same dataset, same tests

### Dataset (identical for both)

- **Current (v9 rebalanced):** `TRAINING_DATA` 33 classes ×4 =132 (32 original expanded 2→4 + `stock_take_away` 4) + synthetic 5/class via `all_templates` =165 → **297 total**. Same `all_templates` in `student_experiment.py` (now includes `stock_take_away`).
- **Gold traces (M2b, optional):** `student_trace.generate_trace` for Qwen `question → trace → answer` verifiable; TF-IDF stays `question → pattern`.

### Routers (identical interface)

```python
# Both expose predict(question) -> (pattern, conf) and predict_with_args
TFIDFRouter:  TfidfVectorizer(1-2, 800 feats) + LogReg, pickle ~240KB, THRESHOLD 0.07
QwenRouter:   Qwen2.5-0.5B-Instruct-4bit + LoRA r=8 alpha=16, <100MB LoRA, <500ms
              input:  "Question: ...\nPattern:"  output: '{"pattern": "...", "args": {...}}'
              fallback: if conf <0.07 or args invalid → regex (same as TF-IDF)
```

### Evaluation harness (same scorer)

`student_experiment.py --dual` (or `validate_dual.py`) runs **both** on:

| Suite | n | Measures |
|---|---|---|
| 120 closed | 120 | routing vs regex, exact answer via `resolve_*` (`check_answer`) |
| 20 novel (fixed) | 20 | never-seen numbers, distinct wording (handshake 47 etc.) |
| 60 human batch60 | 60 | real human wording, includes 6 take_away |
| 100 breaking | 100 | 15 stock outside 33 patterns — stress test |

Metrics side-by-side per router: `train_acc / CV 3-fold / routing 36 / novel 20/20 / human 60 / breaking 100 / size / lat_ms`. Saved to `results_dual.json` for thesis.

### Training

- **TF-IDF:** `student_router.train()` (sklearn, <1s, 800 feats, CV 0.96 on 132).
- **Qwen LoRA (MLX):** `qwen_lora_finetune.py`:
  - Base `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (~600MB, fits 8GB alone)
  - Data: same 297 JSONL `{"prompt": "Question: ...\nPattern:", "completion": '{"pattern": "handshake", "args": {"n": 47}}'}`
  - LoRA r=8, lr 2e-4, 3 epochs, batch 4, dropout 0.05 — via `mlx_lm.lora`
  - Verification: each trace/step passes `verify_trace` / executor before counting as correct
  - If 8GB insufficient for local training, keep TF-IDF as evaluated router and leave Qwen as few-shot placeholder — honest limits.

## Implementation plan (iterative, no 8GB breakage)

1. **Rebalance (done 2026-08-31):** add `stock_take_away` to `TRAINING_DATA` + `all_templates`, retrain TF-IDF to 4/class (132, CV 0.962), verify `60/60` and `20/20` without regression. Validates condition B (student learns take_away).
2. **Scaffold dual:** `tools/qwen_router.py` (same `predict` signature, returns `not_trained` until LoRA exists) + `student_experiment.py --dual` placeholder table. TF-IDF real vs Qwen placeholder.
3. **Qwen real:** implement MLX load + LoRA training via `qwen_lora_finetune.py` following `student_trace` gold traces. Train and measure `<500ms`.
4. **Thesis:** dual table in RESULTS.md + figure `dual_router.png` (TF-IDF vs Qwen on 120/20/60/100).

## Honest expectations

- TF-IDF will keep winning on **lat/size** (240KB vs ~100MB, 0.26ms vs ~300ms) and on CV with small data.
- Qwen should win on **100 breaking** and on paraphrases like "heavier/kilogram" where TF-IDF falls to `fraction_compare`.
- If Qwen does not win, that is an honest finding: TF-IDF already learns structure for this domain — report it as such.

## Status

- Step 1 done: 33×4 =132, 297 with synthetic, CV 0.962, `stock_take_away` now learned (0.27), `same_weight` kilogram correct, `e-r-11` disambiguated.
- Step 2 done: `qwen_router.py` scaffold + `student_experiment.py` dual note.
- Next: train Qwen LoRA when RAM/time window allows.
