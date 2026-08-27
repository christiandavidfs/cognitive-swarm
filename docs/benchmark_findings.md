# Leveled Benchmark Results (raw notes)

Config: 36 problems (12 easy / 12 medium / 12 hard), 3 categories (math/reasoning/code).
Local models use optimized prompts. Accuracy = problem solved if correct in the run.

## Accuracy by mode
mode      overall  easy  medium  hard   cat-math  cat-reason  cat-code
phi       28%      67%   8%      8%     42%       25%         17%
qwen      28%      33%   42%     8%     17%       33%         33%
debate    39%      42%   33%     42%    67%       42%         8%
api       8%       8%    8%      8%     8%        0%          17%

## Value-add of debate
- Problems where NEITHER single model solved but DEBATE solved: 7/36
- Debate solved total: 14/36
- Both singles solved: 3/36
- Union (oracle) phi OR qwen: 17/36 = 47%
- Debate: 14/36 = 39%  (3 fewer than oracle union)

## Key observations
1. Debate beats both single small models (39% vs 28%).
2. Debate beats large API model (39% vs 8%). API (Qwen 27B) is crippled by
   persistent "thinking mode" output that pollutes answers regardless of prompt.
3. Debate adds most value on HARD problems: 42% vs 8% for either single model.
4. Debate weakest on CODE (8%) — consensus picks wrong on code.
5. Debate solved 7 problems no single model solved, but lost 3 that singles got.
   The consensus engine is not yet optimal (oracle union would be 47%).
6. Small models on easy problems via optimized prompts are strong (phi 67% easy).
