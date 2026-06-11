# Model Selection Benchmark v1 — Summary

> All numbers from real runs on RTX 5090 32 GB. No placeholders.
> Judge: **qwen3:32b** (not a competitor — see DECISIONS.md ADR for rationale).
> Cross-validation judge: **gemma4:31b** on 20 item subset.

## Quality / Latency / Resource (three-dimension table)

| Model | Params | Quant | Faithfulness | Hit Rate | n | Judge | Cold TPS | Warm TPS | Cold Wall(s) | VRAM Δ(MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| elyza-jp-8b | 8.0B | Q4_K_M | 1.0000 | 0.9000 | 20 | qwen3:32b | 288.89 | 280.04 | 1.63 | 6347 |
| gemma4-31b | 31.3B | Q4_K_M | 1.0000 | 0.9500 | 20 | qwen3:32b | 70.45 | 70.66 | 5.62 | 19690 |
| nemotron-nano-9b-jp | 8.9B | Q4_K_M | 0.9792 | 1.0000 | 20 | qwen3:32b | 190.72 | 189.52 | 7.67 | 11363 |
| swallow-8b | 8.0B | Q8_0 | 1.0000 | 1.0000 | 20 | qwen3:32b | 169.27 | 190.64 | 6.94 | 9547 |

> faithfulness = fraction of answer claims entailed by provided context (judge-based, 1 run).
> hit_rate = fraction of items where answer matches reference answer (judge-based).

## Judge Agreement (cross-validation)

- Primary judge: **qwen3:32b**
- Cross judge:   **gemma4:31b** (also a competitor; cross-judges **only to validate primary judge reliability**, not to score itself)
- Subset size:   20 items (random seed 42)
- Hit agreement rate: **1.0**
- Cohen's κ (hit): **1.0**
- Faithfulness agreement rate (|Δ|<0.2): **0.95**
- Disagreements: 0 items

## Constraint → Recommended model (decision table)

| Constraint | Recommended | Rationale |
|---|---|---|
| VRAM ≤ 10 GB + best Japanese quality | **swallow-8b** | 8B models fit 5–10 GB; winner by hit_rate |
| VRAM ≤ 10 GB + fastest tokens/s | **elyza-jp-8b** | highest warm TPS among ≤10 GB models |
| VRAM ≤ 20 GB + highest quality | **nemotron-nano-9b-jp** | highest hit_rate across all models; 31B needs ~20 GB |
| Fastest cold start (lowest TTFT) | **elyza-jp-8b** | smallest cold_start_wall_s |
| Multilingual (JP + EN) required | **gemma4-31b** | natively multilingual; 8B JP fine-tunes may degrade EN |
| Apache 2.0 license required | **gemma4-31b** | Apache 2.0; verify 8B model licenses before production |

## Protocol integrity notes

- **qwen3:32b** judges all competitors. It is NOT a competitor (deployment/content layer separation — see DECISIONS.md).
- **gemma4:31b** cross-judges a 20-item subset to validate primary judge reliability. Its own main scores are judged by qwen3:32b, never itself.
- All latency and VRAM numbers from live Ollama runs on RTX 5090 32 GB.
- Golden set: 20 context-grounded QA items, neutral tech/Japanese knowledge, no customer data.
- Golden set source: golden_qa.jsonl (committed to repo).

