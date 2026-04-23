# Case Studies

*End-to-end breakdowns of milestone models and training systems — how architecture, algorithms, infrastructure, and evaluation combine in real production work.*

---

## What This Is

Topic folders explain *individual mechanisms*. Case studies explain *how those mechanisms fit together* in a specific system. They're the integration layer of the repo.

Each case study links out to concept pages for the mechanisms it uses rather than re-explaining them. If you find yourself writing the same mechanism twice, extract it into a concept page and link from both case studies.

---

## What Belongs Here

- **Milestone models** — end-to-end breakdowns of papers/releases that defined a line (GPT-3, LLaMA 3, DeepSeek-V3, Claude, Gemini, etc.).
- **Training systems** — production infrastructures that are instructive in themselves (Composer 2, Megatron-LM pipelines, DeepSpeed setups).
- **Reasoning systems** — long-CoT training pipelines like DeepSeek-R1 and o1-style reports.

## Reading Order

Pick whichever case study matches a concept area you're studying — case studies are lookups, not a linear path.

---

## Case Studies

- [Composer 2](composer2.md) — MosaicML's pretraining stack.
- [OLMo 2](olmo-2.md) — AI2's fully-open 7B/13B/32B reproducibility reference.
- [DeepSeek-V3](deepseek-v3.md) — 671B/37B MoE at GPT-4o parity for ~$5.6M.
- [DeepSeek-R1](deepseek-r1.md) — the same V3 base pushed to o1-1217 reasoning parity via long-CoT RL; includes R1-Zero (RL from base, no SFT) and the 4-stage production pipeline.
- [Kimi k1.5](kimi-k1-5.md) — Moonshot's long-CoT reasoning model, contemporaneous with R1. Different recipe (online policy mirror descent, length penalty, partial rollouts, long2short distillation), multimodal, not open-weights. Pair with DeepSeek-R1 to see two design points for long-CoT RL.

---

## Related

- Every other folder — case studies link back to concept pages across the repo.
