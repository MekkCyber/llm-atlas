# MOPD — Cross-Modal Multi-Teacher On-Policy Distillation

*Depth — combining many domain-RL teachers into one multimodal generalist via overlap-set distillation.*

**TL;DR:** Train K specialist teachers, one per domain (math, code, OCR, video grounding, tool use, …), each with its own RL recipe. Then distill all of them into a single student multimodal model: the student generates an on-policy rollout, a router picks the relevant teacher for that prompt, and the teacher provides token-level logit feedback. The trick is computing the advantage only on the *Top-k overlap* between teacher and student distributions, scaled to down-weight formatting tokens — keeping the teacher's signal where both agree it's plausible and ignoring it elsewhere. Used in Keye-VL-2.0 (2026) to combine 13 teachers into a single MoE generalist.

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md), [rlvr](rlvr.md)
**Related:** [_rewards](_rewards.md) · [video-rl](video-rl.md)

---

## What it is

Modern multimodal models are pushed in many directions at once: math reasoning, code, OCR, counting, video grounding, dense captioning, tool use, search. A single multi-task SFT/RL run dilutes signal across domains and triggers catastrophic forgetting between updates. MOPD's alternative: train one *small* teacher per domain (each is RL-tuned to be very good at its own task), then distill all of them back into a single student.

The novelty isn't multi-teacher distillation per se — it's the *on-policy + overlap-set + token-category-aware* combination that makes it work without the student being pulled apart by conflicting teachers.

## How it works

### Train K domain teachers

Each teacher $T_i$ is RL-trained on its own domain with its own reward function. Keye-VL-2.0 uses 13 teachers across math, code, OCR, grounding, counting, video, tool-use, and others.

### Per-prompt routing

For each training prompt $x$, a domain router (a lightweight classifier) picks which teacher $T_i$ to use. Off-domain teachers are ignored on this prompt — they don't get to push the student in irrelevant directions.

### On-policy student rollout

The student $S$ generates a response $y \sim S(\cdot \mid x)$. The teacher $T_i$ does *not* generate; it only provides token-level distributions $T_i(\cdot \mid x, y_{<t})$ on the student's own tokens.

### Top-k overlap set

For each token position $t$:

$$ \mathcal{O}_t = \mathrm{Top}_k(S(\cdot \mid x, y_{<t})) \cap \mathrm{Top}_k(T_i(\cdot \mid x, y_{<t})) $$

The overlap set is the small subset of tokens that *both* student and teacher consider plausible. Advantage / loss is computed only over $\mathcal{O}_t$:

$$ L_t \propto \sum_{v \in \mathcal{O}_t} (\,\text{teacher signal at } v\,) \cdot (\,\text{student-action prob at } v\,) $$

This is the key restraint: when the teacher and student fundamentally disagree (no overlap), the teacher's signal is ignored — the student isn't yanked toward a token it considers implausible.

### Token-category-aware advantage scaling

Different tokens have different roles in a generation: content tokens carry the answer, formatting tokens carry structure ("```python", "Answer:", "<think>"). MOPD down-weights formatting tokens so the teacher's content signal dominates:

$$ \tilde{L}_t = \gamma_{\text{cat}(t)} \cdot L_t $$

where $\gamma_{\text{format}} < \gamma_{\text{content}}$. Avoids the failure mode of the student blindly imitating the teacher's formatting quirks.

### Update

Apply the weighted loss with a standard policy-gradient-style update. The student's weights move; the teachers stay frozen.

## Why it matters

- **Combines specialists into a generalist without RL on the combined task.** RL'ing one model on 13 mixed rewards is brittle (reward interference, schedule sensitivity). Specializing 13 teachers and then distilling is more tractable.
- **Preserves multimodal capability while injecting domain skill.** The on-policy + overlap-set design prevents the teacher from corrupting the student's general behaviour.
- **Cheaper inference.** Production deploys only the student, not the 13 teachers.
- **Scales naturally with new domains.** Add a teacher, the router picks it on its prompts, distill — no retraining of existing teachers or student-from-scratch.

## Gotchas & tricks

- **Router quality bounds everything.** Mis-routing pulls the student toward an off-domain teacher and degrades the relevant capability. The router needs to be calibrated across both unambiguous prompts and edge cases.
- **Overlap set can be empty.** When student and teacher fundamentally disagree, $|\mathcal{O}_t| = 0$ and there's no gradient. Some teachers will have many empty-overlap tokens early in training; tolerate this — the student will move toward the teacher elsewhere and the overlap grows.
- **Token-category labelling matters.** The format-vs-content split needs to be reliable. Keye-VL-2.0 uses a rule + classifier hybrid; bad labelling leaks teacher quirks back in.
- **Doesn't replace RL.** MOPD distills teachers that themselves were RL'd. If your task lacks a tractable reward, you can't make a teacher in the first place.
- **Teacher diversity matters.** Two teachers with very similar capabilities don't add much over one. Pick teachers whose domains are genuinely orthogonal.

## Sources

- Paper: *Kwai Keye-VL-2.0 Technical Report* — Wen et al., Kwai/Kuaishou, 2026 — [arXiv 2606.10651](https://arxiv.org/abs/2606.10651).
- Background: on-policy distillation lineage (DAGGER and successors); knowledge distillation foundations.
