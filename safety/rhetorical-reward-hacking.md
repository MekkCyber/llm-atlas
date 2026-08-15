# Rhetorical Reward-Hacking
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** LLM-as-judge reviewers can be **reward-hacked by rhetoric alone** — rewriting the surface of a paper while preserving its scientific content shifts assessment scores. Evidence framing and novelty stance produce the largest positive-negative contrasts. Stricter review protocols lower absolute scores but do not reduce rhetorical sensitivity.

**Prereqs:** [_attacks](_attacks.md), [../post-training/cot-reward-model](../post-training/cot-reward-model.md)
**Related:** [../post-training/_rewards](../post-training/_rewards.md), [../evaluation/README](../evaluation/README.md)

---

## What it is

LLMs are increasingly deployed as **judges** — in RLHF/RLAIF reward loops, in pairwise-preference eval, and now in AI-based peer review. A judge that scores differently based on **how** claims are phrased (rather than **what** is claimed) is a compromised optimizer target: whatever policy is trained against it will preferentially learn to write in the judge-preferred style, whether or not the underlying content improves.

Rhetorical reward-hacking is the concrete failure mode: **surface rewrites that preserve content shift the judge's score**, in a direction that generalizes across reviewer instances. The Li et al. (2026) study operationalizes this on academic peer review.

## How it works

**Experimental rig.**

- **4,200 ICLR 2026 submissions** as the base corpus.
- **LLM rewriter** modifies six rhetorical dimensions per paper, one at a time and jointly:
  1. **Evidence framing** (e.g. "our results show X" vs. "we observe X in the specific settings tested").
  2. **Novelty stance** (bold vs. modest claims of contribution).
  3. **Tone** (assertive vs. hedged).
  4. **Hedging**.
  5. **Structure** (section flow, transition style).
  6. **Prior-work positioning** (relative to competitors).
- **Content preservation constraint** — the scientific claims, experiments, and citations must stay intact. Rewrites are checked for content preservation.
- **Five LLM reviewers** score the original vs. rewritten versions under multiple review protocols (standard vs. stricter).

**Attack signal.** The score delta $\Delta = \text{score}(\text{rewritten}) - \text{score}(\text{original})$ measures the judge's rhetorical sensitivity per dimension.

## Why it matters

- **Evidence framing and novelty stance** produce the largest positive-negative contrasts — a paper's *stated confidence in its contribution* moves review scores substantially without moving the science.
- Score changes depend on the **reviewer's initial assessment**: rhetorical rewrites push weak reviewers more than strong ones.
- **Stricter review protocols reduce overall scores by ~1.36 points** on average but **do not substantially reduce rhetorical sensitivity**. Sternness alone doesn't fix it.
- This is a general fact about LLM judges, not a peer-review-specific one. Any RL loop using an LLM judge has this exploit surface — a policy trained against it will learn the rhetorical patterns that raise scores.

## Gotchas & tricks

- **Content-preserving rewrites are surprisingly easy.** LLMs can rewrite a paragraph in dozens of stylistic registers while keeping claims intact. Reward hacking via style is cheap.
- **Protocol strictness isn't a defense.** The scoring distribution shifts, but the rhetorical-attack gradient is preserved. Real defenses need to look at rewrite-invariance (test whether the score changes under content-preserving paraphrases).
- **Attack scales with judge capability, not against it.** Stronger judges have finer rhetorical sensitivity — they are more, not less, exploitable in absolute terms.
- **Aggregation across many judges partially helps.** Independent-error averaging reduces variance, but rhetorical sensitivity is a shared prior across LLM judges (all trained on similar data), so aggregate rewrites still shift the mean.
- **Detection possible via rewrite-consistency probes.** Run the same paper through $k$ rhetorical rewrites, look at score variance. High variance → the judge is being reward-hackable in real time.

## Sources

- Paper: *How Can Rhetoric Reward-Hack AI Reviewers? Dissecting Rhetorical Sensitivity in AI-Based Peer Review* — Ming Li, Chenguang Wang, Xirui Li, Xinyue Zeng, Dianqi Li, Peng Shi, Dawei Zhou, Tianyi Zhou, 2026 — [arXiv:2608.08975](https://arxiv.org/abs/2608.08975).
