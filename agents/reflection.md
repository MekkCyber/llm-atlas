# Reflection (agent capability)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reflection is the agent capability of **producing a revised solution that beats the previous attempt**, given the previous attempt and its verdict. AREX-2 (Qian et al., 2026) argues reflection is domain-agnostic and can be trained from synthetic ML and algorithmic-coding trajectories where verifiable feedback is available, then transferred to open-ended tasks (deep research, QA, GAIA) without further fine-tuning. Paired with long-horizon execution, reflection is the backbone capability of modern self-improving agents.

**Prereqs:** [_self-evolving-agents.md](./_self-evolving-agents.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md).
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md).

---

## What it is

An agent gets `(task, previous_attempt, verdict)` and must return a *better* attempt. The capability is the delta between round $k$ and $k+1$:

$$
\Delta_k = \text{quality}(a_{k+1}) - \text{quality}(a_k)
$$

Reflection is "trained" when $\Delta_k > 0$ in expectation and stays positive for many rounds (so the agent doesn't just move laterally after one revision).

The companion capability is **long-horizon execution**: $\Delta_k$ stays positive at $k=20, 50, \ldots$, not just $k=2$.

---

## How it works

**Training data (AREX-2 recipe).** Pick domains with cheap, verifiable feedback:
- **ML tasks** (MLE-bench-flavored): train a classifier, submit prediction, get a score.
- **Algorithmic programming**: unit tests give verdicts.

Synthesize long-horizon improvement trajectories:

1. Start with a reasonable attempt.
2. Run verifier; log score + stderr.
3. Prompt the agent to revise given `(attempt, score, stderr)`.
4. Repeat for $N$ rounds; keep trajectories where $\Delta_k > 0$ for most $k$.

Train the agent on these trajectories (SFT, or RL with a per-round reward on $\Delta_k$).

**Transfer.** Trained agents reflect usefully on **unseen** domains — deep research (BrowseComp, DeepSearchQA), general QA (HLE), tool-using (GAIA) — without additional fine-tuning. That's the domain-agnostic claim: reflection as a *skill* lives above any particular task's semantics.

---

## Why it matters

- **Decouples agent capability from task-specific training.** You pay for ML + coding corpora once; the resulting agent reflects well across tasks.
- **AREX-2 numbers (Qwen3.8-27B backbone):** 81.8 on MLE-bench Lite, 70.7 on Frontier-CS; transfers to 84.0 BrowseComp, 52.6 HLE, 92.2 GAIA, 93.8 DeepSearchQA.
- **Scales with budget.** Performance keeps improving as the round budget grows — reflection is a knob, not a one-shot capability.
- **Prerequisite for recursive self-improvement.** RSI needs a working reflection step; AREX-2 is a candidate production of that step.

---

## Gotchas & tricks

- **Need verifiable feedback at training time.** If the verifier is a learned RM, you can inherit its biases. The AREX-2 recipe explicitly picks domains where the verifier is a *program* (unit tests, scoring harness).
- **Long-horizon collapse.** Reflection quality often drops after $k \sim 5$–$10$ for naive models — later revisions start regressing on earlier gains. Explicit long-horizon training (keep trajectories that stay monotone to large $k$) is the fix.
- **Diversity of mistakes matters.** Trajectories that always fail in the same way teach the agent one correction, not reflection. Mix verifier flavors (unit tests, metric scores, lint, runtime) to force general-purpose correction.
- **Reflection $\ne$ self-critique.** Self-critique produces a *judgment* about an answer; reflection produces a *revised answer*. Many papers conflate them; keep them separate when designing training signals.

---

## Sources

- Paper: *AREX-2: Advancing Self-Improving Agents through Long-Horizon Reflective Tasks* — Qian, Li, Luo, Wei, Chen, Lu, Hu, Xiao, Wang, Li, Ye, Dou, Lian, Liu, 2026. arXiv:2609.38288. BAAI.
