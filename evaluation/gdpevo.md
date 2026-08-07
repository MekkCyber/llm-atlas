# GDPevo — Evolution-Native Agent Benchmark

*Depth — a benchmark design where held-out test tasks *recombine* atomic business rules from training tasks, making self-evolution gains causally attributable.*

**TL;DR:** Evaluating agent *self-evolution* (agents that update persistent state from prior experience) is hard: existing benchmarks don't attribute test-time gains to training experience, remain vulnerable to contamination, and cover economically thin domains. GDPevo (Liu et al., 2026, PrismShadow / NYU) grounds evaluation in **GDP-related enterprise workflows** (CRM, ERP, finance, healthcare, legal, data-centric) and uses **rule hybridization** — decompose each workflow into atomic business rules, distribute subsets across training tasks, recombine unseen subsets in held-out tests. V1: 120 tasks in 12 groups (5 train + 5 test each). Fully automated pipeline expands to 240 tasks in 24 groups within two days. Best self-evolution gain: **+16.44 pp** on held-out; oracle ceiling **91.6%**.

**Prereqs:** [README.md](README.md)
**Related:** [../data/decontamination.md](../data/decontamination.md) · [../agents/agent-harness.md](../agents/agent-harness.md) · [../agents/skill-kd.md](../agents/skill-kd.md)

---

## What it is

An evolution-native benchmark: designed so that gains from a self-evolving agent (one that reads its own prior experience and updates persistent state) can be *causally* attributed to training experience rather than contamination or memorization. Six enterprise workflow domains: CRM, ERP, finance, healthcare, legal, data-centric.

## How it works — rule hybridization

1. **Atomic rules.** Each enterprise workflow is decomposed into a set of small, composable business rules (e.g. "approve refund if amount < $50 and reason ∈ approved-list").
2. **Rule assignment to training tasks.** For each group of 12, distribute rule subsets across 5 training tasks — each training task uses a specific rule subset.
3. **Rule recombination in test tasks.** The 5 held-out test tasks in the same group recombine rules *from* the training subsets in configurations the agent has not seen. Solving them requires having generalized the rules from training experience.
4. **Automated regeneration.** The full pipeline runs unsupervised; V2 (240 tasks / 24 groups) is generated within two days. This is the paper's answer to contamination — regenerate the benchmark faster than models can be re-trained.

Four agent supervision types are compared: (harness, model) × {passive, reactive, memory-driven, memory + reflection}, so the framework isolates which self-evolution mechanism matters.

## Why it matters

- **First contamination-resistant self-evolution benchmark.** Rule hybridization makes test-time gains attributable *by construction* — the agent has to compose rules it only saw individually.
- **Auto-regeneration as a contamination policy.** V1→V2 in two days means the benchmark can outpace training-data leakage cycles; more benchmarks should adopt this.
- **Real economic domains.** GDP-related workflows are what agent products are actually deployed for, so score deltas here transfer to product value.
- **Names the self-evolution gap.** +16.44 pp is real but the oracle ceiling is 91.6% — self-evolution *works* but is nowhere near capacity.

## Gotchas & tricks

- **Rule granularity is the design lever.** Too coarse and recombination reduces to memorization; too fine and rules become noisy or fragmentary. The paper's decomposition is domain-tuned.
- **Auto-regeneration doesn't cover targeted contamination.** If a red-teamer specifically ingests the V1 tasks, V2 is safe but V1 scores are polluted. Report on the *current* version.
- **Supervision-type comparisons need matched budgets.** Memory-driven agents look better partly because they see more effective context — control for token / step budget when comparing.
- **Rule hybridization pattern generalizes** to any domain with composable rules (tool APIs, spec compliance, code refactoring guidelines). The GDP framing is the initial application.

## Sources

- Paper: *GDPevo: Evaluating Agent Self-Evolution on Real Business Tasks* — Liu, Qu, Liu, Liu, Yu, Xu, Wu, Qian, Chen, Zheng, Hu, 2026 — [arXiv 2608.03764](https://arxiv.org/abs/2608.03764). PrismShadow / New York University. Pipeline release: [github.com/Prism-Shadow/GDPevo](https://github.com/Prism-Shadow/GDPevo).
