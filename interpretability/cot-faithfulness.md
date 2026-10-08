# CoT Faithfulness

*Depth — the measurement problem of whether a model's chain-of-thought actually reflects the computation that produced its answer.*

**TL;DR:** A model's chain-of-thought (CoT) is a *claim* about how it reached its answer, not evidence. Faithfulness is the degree to which that claim is true — the CoT content actually drives the answer, and the strategies verbalized in it match the strategies interpretability tools detect internally. Lanham (2023) showed CoT faithfulness is partial and gets worse with scale; CIA (2026) ties it directly to interpretability-detected strategies and shows it can be trained for.

**Prereqs:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md)
**Related:** [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md), [../safety/_scheming.md](../safety/_scheming.md)

---

## What it is

Three different things get called "CoT faithfulness" in the literature, and keeping them separate matters:

- **Causal faithfulness** (Lanham 2023): if you intervene on the CoT (truncate it, add errors, paraphrase it), does the answer change? If yes, the CoT is at least partly causally driving the answer.
- **Strategy faithfulness** (CIA 2026): the strategy the CoT names ("I'll use long division") matches the strategy that probes or activation steering detect in the model's internals.
- **Externalization faithfulness** (Apollo 2024, see [../safety/cot-monitoring.md](../safety/cot-monitoring.md)): the CoT *verbalizes* the reasoning the model actually did — including inconvenient reasoning like scheming or sandbagging intent.

All three can fail independently. A CoT can be causally faithful (truncation hurts answers) but strategy-unfaithful (the written strategy is a cover story). It can be externalization-faithful on benign reasoning but hide scheming reasoning.

## How it works

### Measuring causal faithfulness (Lanham)

Intervene on the CoT at generation time:

```
baseline:         prompt → CoT → answer
intervention A:   prompt → CoT with mistake → answer       (does the answer change?)
intervention B:   prompt → CoT paraphrased → answer         (does the answer change?)
intervention C:   prompt → CoT truncated to k steps → answer
```

Faithfulness = fraction of (prompt, intervention) pairs where the answer changes. Higher = CoT is doing load-bearing work. Findings: it's variable across tasks; *larger* models often have *less* faithful CoT (they answer from internal priors regardless of what the CoT says).

### Measuring strategy faithfulness (CIA)

Pick a task with a small, enumerable strategy space (e.g., three distinct approaches to a math problem). For each strategy:

1. Train a probe or activation-steering vector that fires when the model is internally running that strategy.
2. For a given rollout, run the probe to detect which strategy the internals say is active.
3. Have an LLM-judge read the CoT and extract which strategy the CoT *claims* is active.
4. **CIA = agreement rate** between the probe-detected and CoT-claimed strategies.

CIA numbers from the paper: three tasks across three LLMs show initial CoT–Interpretability Alignment between **44.8% and 75.9%**, i.e. between "coin-flip" and "mostly aligned."

### Training for strategy faithfulness

CIA (2026) adds the strategy-agreement score as a *reward* in post-training. The model is RL-trained to make its CoT match its internals. The paper reports "substantial improvement" in CoT parametric faithfulness while preserving task accuracy — i.e. the model doesn't game accuracy away by inventing cover-story CoTs that happen to agree with internals.

## Why it matters

- **CoT monitoring assumes faithfulness.** The whole premise of reading the CoT to detect misbehavior is that the CoT reflects what the model is doing. Unfaithful CoT means the monitor reads a cover story. See [../safety/cot-monitoring.md](../safety/cot-monitoring.md).
- **Faithfulness is trainable.** This is the key implication of CIA: faithfulness isn't a fixed property of a model, it can be a training objective. Which means safety work can treat "faithful CoT" as a design target, not a hoped-for property.
- **Interpretability enters post-training.** Using a probe or steering vector as a reward signal is a new post-training ingredient — the interpretability tool is in the training loop, not just the eval loop.

## Gotchas & tricks

- **CIA requires a small strategy space.** If you can't enumerate the strategies, you can't build probes for them. On open-ended generation this is open research.
- **Training for faithfulness could encourage the model to notice the probe.** If the probe becomes part of the training signal, the model might learn to produce internals that trigger the probe without actually using that strategy — a Goodhart's-law risk. Mitigation: rotate probes, hold some out as evaluators only.
- **The three faithfulness notions can be mutually incompatible.** Pushing hard on strategy faithfulness might reduce externalization faithfulness if the easiest way to match the probe is to produce a terse strategy-name instead of full verbalized reasoning.
- **Base-rate baseline matters.** On a 3-strategy task, chance CIA is 33%. The reported 44.8–75.9% needs this context: some of it is above-chance alignment, some of it is the probe being noisy.

## Sources

- Paper: *Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment* — Ravfogel, Zhao, Choi, NYU + NYU Shanghai, 2026 — [arXiv:2609.38972](https://arxiv.org/abs/2609.38972).
- Paper: *Measuring Faithfulness in Chain-of-Thought Reasoning* — Lanham et al., Anthropic, 2023 — [arXiv:2307.13702](https://arxiv.org/abs/2307.13702).
- Paper: *Reasoning Models Don't Always Say What They Think* — Chen, Benton et al., Anthropic, 2025.
- Related: *Frontier Models are Capable of In-context Scheming* — Meinke et al., Apollo, 2024.
