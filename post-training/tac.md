# TAC — Transfer-Aware Curriculum for Multi-Domain RLVR

*Depth — a bandit curriculum for multi-domain RLVR that biases sampling toward domains whose gradient updates help the rest.*

**TL;DR:** Multi-domain RLVR (math + code + science + …) needs a **curriculum**: how often to sample each domain per RL step. Prior learnability-based bandits track "where the policy is currently improving" but ignore whether a step on domain A also helps domains B, C. **TAC** combines per-domain **advantages** (local learnability) with projected **gradient alignment** across domains (cross-domain transfer) as its bandit signal, both computed from information GRPO produces anyway. Beats learnability-only bandits by up to **2.8 points (10% relative)** on a six-domain reasoning suite with <1% wall-clock overhead.

**Prereqs:** [rlvr.md](rlvr.md), [grpo.md](grpo.md)
**Related:** [rl-prompt-curation.md](rl-prompt-curation.md), [_rl.md](_rl.md), [_post-training.md](_post-training.md)

---

## What it is

An online domain-sampler for multi-domain RLVR pipelines. TAC replaces static / hand-tuned / learnability-only schedules with a bandit whose reward signal explicitly captures **cross-domain transfer**. The core insight: a GRPO update is a gradient; the gradient's alignment with the gradients of *other* domains directly measures whether stepping on this domain will help or hurt them.

## How it works

### Two signals

For each domain $d \in \{1, \ldots, D\}$, TAC maintains:

- **Advantage-based learnability** $a_d$ — the mean per-prompt advantage magnitude in recent GRPO rollouts for domain $d$. If advantages are near zero, the policy is either saturated or unable to make progress on $d$.
- **Cross-domain transfer** $t_d$ — the average alignment (cosine or inner product) between the GRPO gradient on domain $d$ and the gradients on the other domains. Positive alignment = stepping on $d$ helps the rest; negative = it hurts.

### Bandit update

Domain $d$'s bandit score combines the two, e.g.

$$
s_d = a_d \cdot (1 + \alpha \cdot t_d)
$$

Sampling probabilities are a softmax over $\{s_d\}$. Domains that are learnable *and* transfer widely get more of the budget; domains that are learnable but hurt other domains get down-weighted.

### Why gradient alignment is nearly free

TAC computes the alignment from **projected gradients** of the GRPO step that is being taken anyway. There is no extra rollout, no extra forward pass, no Hessian-vector product. The paper reports <1% wall-clock overhead vs learnability-only bandits.

## Why it matters

- **Multi-domain RLVR is the actual regime.** Frontier reasoning models train on mixtures — math + code + science + logic + medicine + tool use. Any curriculum that only optimizes "where I am improving" over-commits to one domain and neglects transfer.
- **Uses signals already computed.** TAC doesn't ask you to train a new predictor or estimate anything expensive. The learnability and gradient signals fall out of GRPO for free.
- **Empirically robust.** Ablations show removing the transfer term collapses performance; TAC is robust on **imbalanced training mixtures** where learnability-only bandits over-commit to the dominant domain.

## Gotchas & tricks

- **Gradient alignment is noisy at small batch.** With few prompts per domain per step, the per-domain gradient estimate is high variance and the alignment sign flips. Smooth over recent steps (EMA).
- **$\alpha$ trades exploration for transfer.** High $\alpha$ starves any domain that isn't currently transferring positively (which locks in an early-training snapshot). Anneal it down over training.
- **Transfer sign flips over training.** Early on, math might help code; late in training the two might compete. TAC's online nature handles this, but you should still monitor per-domain accuracy on held-out sets to catch pathological schedules.
- **Domain granularity matters.** Splitting math into "arithmetic" and "algebra" changes the transfer matrix. Pick the smallest domain granularity that has enough prompts per step for a stable gradient estimate.

## Sources

- Paper: *Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR* — Yang, Liu, He, Zhang, Schölkopf, Jin (Toronto / Vector Inst. / CMU / Princeton / UIUC / MPI-IS), 2026 — [arXiv:2606.25178](https://arxiv.org/abs/2606.25178). Reports TAC gains on Qwen3-1.7B and Llama3.2-3B on a six-domain suite.
