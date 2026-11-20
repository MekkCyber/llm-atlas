# Purple-Teaming (Coverage-Aware Adversarial Training)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training recipe where a red-team policy generates *adversarial inputs* and the target model is adversarially trained against them — but with an explicit **coverage reward** (via dictionary learning over attack representations) that forces the red-team to produce *diverse* attacks rather than collapsing onto a few high-success modes. Preserves benign task performance. Introduced in PIPETTE for prompt-injection defense; the pattern generalizes to any adversarial-training setup where mode collapse of the attacker ruins robustness gains.

**Prereqs:** [_attacks.md](_attacks.md), [prompt-injection.md](prompt-injection.md)
**Related:** [_jailbreaks.md](_jailbreaks.md)

---

## What it is

Standard adversarial training for LLM safety: (1) a red-team policy generates attacks maximizing attack-success-rate (ASR); (2) the target model is fine-tuned to resist those attacks. Known failure: the red-team collapses to a handful of attack strategies, the target learns to defend against exactly those, and robustness to attacks outside that mode is unchanged.

Purple-teaming adds a **coverage objective** on top of the red-team's ASR objective: reward attacks that are representable by *different* basis elements in a learned dictionary over attack embeddings.

---

## How it works

Let $\pi_R$ be the red-team policy and $\pi_T$ the target model. Jointly train:

1. **Red-team reward** $R_{\text{red}} = \alpha \cdot \text{ASR}(\pi_R, \pi_T) + \beta \cdot \text{Coverage}(\pi_R)$, where
   $$
   \text{Coverage}(\pi_R) = \mathbb{E}_{a \sim \pi_R} \big[ H(\text{dict-code}(\phi(a))) \big]
   $$
   with $\phi$ an attack embedding and `dict-code` the sparse code under a dictionary learned online over the attack distribution.
2. **Target update.** Fine-tune $\pi_T$ on $(a, y_{\text{benign}})$ pairs, where $y_{\text{benign}}$ is the correct behavior under attack. Preserve benign-task performance with a standard SFT / DPO regularizer on clean data.
3. **Alternating optimization.** Red-team and target update in alternation, both staying on-policy.

The dictionary is updated online as the red-team's attack distribution shifts, keeping the coverage signal relevant.

---

## Why it matters

- **Robustness ≠ high ASR.** Prior red-teaming optimized for ASR and inherited mode collapse. Coverage reframes the objective so the red-team actually probes the space the defender needs to cover.
- **Preserves benign performance.** Adversarial training famously degrades clean-task performance. On-policy adversarial training with benign regularization reportedly keeps near-zero performance loss on benign evals.
- **Transfers across attack classes.** PIPETTE demonstrates the pattern on prompt injection; the recipe (dictionary-coverage reward + on-policy co-training + benign regularizer) applies to jailbreaks and agent misuse with the same shape.

---

## Gotchas & tricks

- **Dictionary quality dominates coverage.** A low-capacity or stale dictionary can't tell diverse attacks apart, and the coverage term degenerates. Online update of the dictionary matters.
- **Benign regularizer is non-optional.** Without it, the target drifts toward a general refusal bias that degrades helpful behavior.
- **Adaptive attacks are the real test.** Static-attack ASR near zero is an easy bar. Adaptive attackers who re-optimize against the trained target are the honest benchmark — PIPETTE reports resilience on AgentDojo and ASB.
- **Not a replacement for system-level defenses.** For prompt injection in particular, purple-teaming raises the bar but trust boundaries and privilege scoping still carry much of the real-world defense.

---

## Sources

- Paper: *PIPETTE: Prompt Injection Purple-Teaming for Robust LLM Agents* — He, Wang, Zhang, Asokan (Waterloo / Zhejiang / KTH) — 2026 — [arXiv:2611.24362](https://arxiv.org/abs/2611.24362).
- Background on red-team mode collapse: Perez et al. 2022 (*Red-Teaming Language Models*), Casper et al. 2023.
- Benchmarks: AgentDojo, ASB.
