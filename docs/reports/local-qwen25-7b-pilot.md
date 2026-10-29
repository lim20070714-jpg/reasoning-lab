# Reasoning strategy evaluation

Verbatim output of `reasoning-lab report --run runs/local-qwen25-7b-pilot`, kept
here because `runs/` is not tracked. Backend: local llama.cpp server,
`Qwen2.5-7B-Instruct-Q4_K_M`, OpenAI-compatible endpoint. The accompanying
`local-qwen25-7b-pilot.config.json` and `.records.jsonl` are the run's own
checkpoint files, so every number below can be recomputed.

Scoring here predates the `\( … \)` answer-wrapper fix shipped in the same
release: the four `m09` rows marked wrong are correct answers the verifier of the
day rejected. Re-scoring the records with the current verifier gives 100.0% for
`cot`, `self_consistency`, `best_of_n` and `mcts`, and leaves `tot` at 83.3%.

## Run

| field | value |
|---|---|
| run id | local-qwen25-7b-pilot |
| jobs evaluated | 60 |
| failed jobs | 0 |
| PRM | `symbolic` |
| seeds | 0 |
| budget per job | max_llm_calls=40 |
| sampling | n_samples=8, branch=3, depth=8 |
| model calls | 874 |
| cache hits | 102 (11.7%) |
| tokens | 343,115 |
| wall time (s) | 19587.2 |

## Dataset

| field | value |
|---|---|
| problems | 12 |
| with reference steps | 12 |
| with flawed variants | 12 |
| subjects | math: 8, physics: 4 |

### Symbolic verifier over the bundled data

| metric | value |
|---|---|
| reference steps | 62 |
| flawed chains | 24 |
| decidable (PASS or FAIL) | 91.9% |
| flawed chains with a rejected step | 100.0% |
| reference steps wrongly rejected | 0.0% |
| rejected steps inside flawed chains | 21.8% |
| chain-level AUROC (reference vs flawed) | 0.993 |
| step-level AUROC | 0.618 |

## Strategies

| strategy | n | accuracy | oracle any | oracle PRM | calls | tokens | tokens/correct | steps | mean score | ECE | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `best_of_n` | 12 | 91.7% | 91.7% | 91.7% | 8.000 | 3021 | 3296 | 6.920 | 0.797 | 0.251 | 0.091 |
| `cot` | 12 | 91.7% | 91.7% | 91.7% | 1.000 | 372 | 406 | 6.830 | 0.708 | 0.209 | 0.545 |
| `mcts` | 12 | 91.7% | 100.0% | 91.7% | 40.000 | 13404 | 14623 | 4.170 | 0.793 | 0.143 | 0.455 |
| `self_consistency` | 12 | 91.7% | 91.7% | 91.7% | 8.000 | 3022 | 3297 | 6.920 | 0.797 | 0.251 | 0.091 |
| `tot` | 12 | 83.3% | 91.7% | 83.3% | 15.830 | 8772 | 10527 | 5.330 | 0.814 | 0.295 | 0.250 |

`oracle any` is the ceiling for best-of-k over the candidate answers a run produced; `oracle PRM` is the accuracy of picking the highest branch-score candidate. Both are recomputed from traces and never consulted by the strategies themselves.

## Answer selection policy

Same recorded candidates, different selectors. A PRM-selected accuracy above the first-answer column is the process reward model paying for itself.

| strategy | n | first | majority vote | PRM mean | PRM max | oracle |
|---|---|---|---|---|---|---|
| `best_of_n` | 12 | 91.7% | 91.7% | 83.3% | 83.3% | 91.7% |
| `cot` | 12 | 91.7% | 91.7% | 91.7% | 91.7% | 91.7% |
| `mcts` | 12 | 75.0% | 83.3% | 91.7% | 91.7% | 100.0% |
| `self_consistency` | 12 | 91.7% | 91.7% | 83.3% | 83.3% | 91.7% |
| `tot` | 12 | 75.0% | 83.3% | 83.3% | 83.3% | 91.7% |

## TTRL-style self-labelling

TTRL replaces the gold answer with a majority vote over the run's own candidates. `majority label acc` is how often that label is actually correct; `reward acc` is how often the implied reward (candidate matches the majority) agrees with the true verdict. Single-chain strategies have a one-vote majority, so read this table on the multi-candidate strategies.

| strategy | problems | majority label acc | reward acc | candidates scored |
|---|---|---|---|---|
| `best_of_n` | 12 | 91.7% | 91.7% | 96 |
| `cot` | 12 | 91.7% | 91.7% | 12 |
| `mcts` | 12 | 83.3% | 85.7% | 84 |
| `self_consistency` | 12 | 91.7% | 91.7% | 96 |
| `tot` | 12 | 83.3% | 88.9% | 27 |

## Calibration · `best_of_n`

| confidence bin | n | avg confidence | accuracy | gap |
|---|---|---|---|---|
| [0.4, 0.5) | 1 | 0.430 | 1.000 | -0.570 |
| [0.5, 0.6) | 2 | 0.541 | 1.000 | -0.459 |
| [0.8, 0.9) | 6 | 0.877 | 1.000 | -0.123 |
| [0.9, 1.0) | 3 | 0.930 | 0.667 | 0.263 |

## Calibration · `cot`

| confidence bin | n | avg confidence | accuracy | gap |
|---|---|---|---|---|
| [0.2, 0.3) | 2 | 0.274 | 1.000 | -0.726 |
| [0.5, 0.6) | 2 | 0.544 | 1.000 | -0.456 |
| [0.8, 0.9) | 7 | 0.849 | 0.857 | -0.008 |
| [0.9, 1.0) | 1 | 0.915 | 1.000 | -0.085 |

## Calibration · `mcts`

| confidence bin | n | avg confidence | accuracy | gap |
|---|---|---|---|---|
| [0.4, 0.5) | 1 | 0.418 | 1.000 | -0.582 |
| [0.5, 0.6) | 2 | 0.572 | 1.000 | -0.428 |
| [0.8, 0.9) | 7 | 0.874 | 0.857 | 0.017 |
| [0.9, 1.0) | 2 | 0.917 | 1.000 | -0.083 |

## Calibration · `self_consistency`

| confidence bin | n | avg confidence | accuracy | gap |
|---|---|---|---|---|
| [0.4, 0.5) | 1 | 0.430 | 1.000 | -0.570 |
| [0.5, 0.6) | 2 | 0.541 | 1.000 | -0.459 |
| [0.8, 0.9) | 6 | 0.877 | 1.000 | -0.123 |
| [0.9, 1.0) | 3 | 0.930 | 0.667 | 0.263 |

## Calibration · `tot`

| confidence bin | n | avg confidence | accuracy | gap |
|---|---|---|---|---|
| [0.4, 0.5) | 2 | 0.436 | 1.000 | -0.564 |
| [0.6, 0.7) | 1 | 0.630 | 1.000 | -0.370 |
| [0.8, 0.9) | 3 | 0.872 | 1.000 | -0.128 |
| [0.9, 1.0) | 6 | 0.942 | 0.667 | 0.276 |

## By subject

| subject | n | accuracy | mean score | calls | tokens |
|---|---|---|---|---|---|
| math | 40 | 87.5% | 0.727 | 14.600 | 5537 |
| physics | 20 | 95.0% | 0.892 | 14.500 | 6082 |

## By level

| level | n | accuracy | mean score | calls | tokens |
|---|---|---|---|---|---|
| algebra | 5 | 100.0% | 0.467 | 15.000 | 5878 |
| arithmetic | 5 | 80.0% | 0.900 | 14.400 | 4556 |
| calculus | 5 | 100.0% | 0.885 | 15.000 | 5138 |
| combinatorics | 5 | 100.0% | 0.662 | 14.400 | 6912 |
| competition | 5 | 100.0% | 0.793 | 14.800 | 4884 |
| electricity | 5 | 100.0% | 0.875 | 14.400 | 5414 |
| geometry | 5 | 20.0% | 0.883 | 14.400 | 4229 |
| mechanics | 5 | 100.0% | 0.892 | 14.800 | 4967 |
| number-theory | 5 | 100.0% | 0.593 | 14.400 | 6063 |
| optics | 5 | 80.0% | 0.880 | 14.400 | 7032 |
| probability | 5 | 100.0% | 0.630 | 14.400 | 6635 |
| thermodynamics | 5 | 100.0% | 0.920 | 14.400 | 6914 |

## Per-problem appendix

| problem | strategy | answer | gold | correct | score | status | steps | calls |
|---|---|---|---|---|---|---|---|---|
| m02 | `best_of_n` | 330 | 330 | yes | 0.897 | unknown | 3 | 8 |
| m02 | `cot` | 330 | 330 | yes | 0.857 | unknown | 5 | 1 |
| m02 | `mcts` | 330 | 330 | yes | 0.923 | unknown | 6 | 40 |
| m02 | `self_consistency` | 330 | 330 | yes | 0.897 | unknown | 3 | 8 |
| m02 | `tot` | 12 | 330 | no | 0.927 | unknown | 7 | 15 |
| m06 | `best_of_n` | 3 | 3 | yes | 0.520 | fail | 11 | 8 |
| m06 | `cot` | 3 | 3 | yes | 0.293 | fail | 5 | 1 |
| m06 | `mcts` | 3 | 3 | yes | 0.565 | fail | 6 | 40 |
| m06 | `self_consistency` | 3 | 3 | yes | 0.520 | fail | 11 | 8 |
| m06 | `tot` | 3 | 3 | yes | 0.436 | fail | 4 | 18 |
| m09 | `best_of_n` | \( 4\pi \) | 4*pi | no | 0.924 | unknown | 4 | 8 |
| m09 | `cot` | \( 4\pi \) | 4*pi | no | 0.831 | unknown | 11 | 1 |
| m09 | `mcts` | 4\pi | 4*pi | no | 0.872 | unknown | 2 | 40 |
| m09 | `self_consistency` | \( 4\pi \) | 4*pi | no | 0.924 | unknown | 4 | 8 |
| m09 | `tot` | 4π | 4*pi | yes | 0.864 | unknown | 3 | 15 |
| m12 | `best_of_n` | 12 | 12 | yes | 0.430 | fail | 12 | 8 |
| m12 | `cot` | 12 | 12 | yes | 0.256 | fail | 7 | 1 |
| m12 | `mcts` | 12 | 12 | yes | 0.897 | unknown | 3 | 40 |
| m12 | `self_consistency` | 12 | 12 | yes | 0.430 | fail | 12 | 8 |
| m12 | `tot` | 12 | 12 | yes | 0.950 | pass | 4 | 15 |
| m14 | `best_of_n` | 56 | 56 | yes | 0.868 | unknown | 14 | 8 |
| m14 | `cot` | 56 | 56 | yes | 0.529 | fail | 12 | 1 |
| m14 | `mcts` | 56 | 56 | yes | 0.418 | fail | 4 | 40 |
| m14 | `self_consistency` | 56 | 56 | yes | 0.868 | unknown | 14 | 8 |
| m14 | `tot` | 56 | 56 | yes | 0.630 | fail | 8 | 15 |
| m17 | `best_of_n` | 0.3456 | 0.3456 | yes | 0.563 | fail | 14 | 8 |
| m17 | `cot` | 0.3456 | 0.3456 | yes | 0.559 | fail | 6 | 1 |
| m17 | `mcts` | 0.3456 | 0.3456 | yes | 0.580 | fail | 7 | 40 |
| m17 | `self_consistency` | 0.3456 | 0.3456 | yes | 0.563 | fail | 14 | 8 |
| m17 | `tot` | 0.3456 | 0.3456 | yes | 0.887 | unknown | 5 | 15 |
| m20 | `best_of_n` | 9 | 9 | yes | 0.897 | unknown | 3 | 8 |
| m20 | `cot` | 9 | 9 | yes | 0.821 | unknown | 9 | 1 |
| m20 | `mcts` | 9 | 9 | yes | 0.872 | unknown | 2 | 40 |
| m20 | `self_consistency` | 9 | 9 | yes | 0.897 | unknown | 3 | 8 |
| m20 | `tot` | 9 | 9 | yes | 0.940 | unknown | 11 | 18 |
| m23 | `best_of_n` | 6 | 6 | yes | 0.887 | unknown | 5 | 8 |
| m23 | `cot` | 6 | 6 | yes | 0.847 | unknown | 9 | 1 |
| m23 | `mcts` | 6 | 6 | yes | 0.910 | unknown | 4 | 40 |
| m23 | `self_consistency` | 6 | 6 | yes | 0.887 | unknown | 5 | 8 |
| m23 | `tot` | 6 | 6 | yes | 0.436 | fail | 4 | 17 |
| p01 | `best_of_n` | 15 | 15 | yes | 0.915 | unknown | 3 | 8 |
| p01 | `cot` | 15 | 15 | yes | 0.915 | unknown | 3 | 1 |
| p01 | `mcts` | 15 | 15 | yes | 0.850 | unknown | 2 | 40 |
| p01 | `self_consistency` | 15 | 15 | yes | 0.915 | unknown | 3 | 8 |
| p01 | `tot` | 15 N | 15 | yes | 0.864 | unknown | 3 | 17 |
| p05 | `best_of_n` | 32 | 32 | yes | 0.856 | unknown | 6 | 8 |
| p05 | `cot` | 32 | 32 | yes | 0.856 | unknown | 6 | 1 |
| p05 | `mcts` | 32 \, \text{W} | 32 | yes | 0.872 | unknown | 2 | 40 |
| p05 | `self_consistency` | 32 | 32 | yes | 0.856 | unknown | 6 | 8 |
| p05 | `tot` | 32 | 32 | yes | 0.937 | unknown | 8 | 15 |
| p09 | `best_of_n` | 300 | 300 | yes | 0.950 | pass | 2 | 8 |
| p09 | `cot` | 300 | 300 | yes | 0.878 | unknown | 3 | 1 |
| p09 | `mcts` | 300 | 300 | yes | 0.871 | unknown | 6 | 40 |
| p09 | `self_consistency` | 300 | 300 | yes | 0.950 | pass | 2 | 8 |
| p09 | `tot` | 300 K | 300 | yes | 0.950 | pass | 4 | 15 |
| p13 | `best_of_n` | 6 | 6 | yes | 0.856 | unknown | 6 | 8 |
| p13 | `cot` | 6 | 6 | yes | 0.856 | unknown | 6 | 1 |
| p13 | `mcts` | 6 | 6 | yes | 0.882 | unknown | 6 | 40 |
| p13 | `self_consistency` | 6 | 6 | yes | 0.856 | unknown | 6 | 8 |
| p13 | `tot` | -6 | 6 | no | 0.950 | pass | 3 | 15 |

## Reproduce

```
uv run reasoning-lab eval --strategies cot,self_consistency,best_of_n,tot,mcts --prm symbolic
uv run reasoning-lab report --run runs/<run-id>
```
