# Delta Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An on-policy distillation variant where the training target is the **log-probability gap between a post-trained teacher and its own base model** — not the teacher's raw distribution. The student is trained to reproduce the *shift* post-training induced in the teacher, not to copy the teacher's base-model habits. Introduced as OPD 2 in the multilingual math reasoning setting; conceptually generalizes anywhere OPD is used.

**Prereqs:** [on-policy-distillation.md](./on-policy-distillation.md).
**Related:** [_post-training.md](./_post-training.md) · [grpo.md](./grpo.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [dpo.md](./dpo.md)

---

## What it is

Standard OPD supervises the student toward the teacher's full next-token distribution $\pi_T$. But $\pi_T = \pi_{T_{\text{base}}} + (\text{post-training shift})$. Matching the full teacher makes the student inherit the teacher's base-model idiosyncrasies — which the student, if it shares a base, already has. Delta distillation isolates the *added* signal.

## How it works

Let $\pi_{T_{\text{base}}}$ be the teacher's base model (pre-post-training) and $\pi_T$ the post-trained teacher. Define the delta signal as the log-probability difference:

$$
\Delta_T(v \mid x, y_{<t}) = \log \pi_T(v \mid x, y_{<t}) - \log \pi_{T_{\text{base}}}(v \mid x, y_{<t})
$$

Positive $\Delta_T(v)$ = post-training made this token more likely; negative = less likely. This is exactly the LM-generation analogue of a DPO reward.

The student is trained so that its own log-probability *shift* from its base tracks the teacher's:

$$
L_{\text{OPD}^2} = \mathbb{E}_{y\sim\pi_S(\cdot\mid x)}\Bigl[\;\sum_t \bigl(\Delta_S(y_t\mid \cdot) - \Delta_T(y_t\mid \cdot)\bigr)^2\;\Bigr]
$$

(other closed-form variants use KL between the softmaxes of $\Delta_T$ and $\Delta_S$.)

Cost: one student forward, one teacher forward, one teacher-base forward per token. The teacher-base pass can be cached per prompt, since it depends only on the input.

## Why it matters

- **Isolates the transferable signal.** The student learns what post-training did, not what the teacher's base was.
- **Cleaner in multilingual and multi-domain settings** where the teacher's base has strong biases (e.g. English preference) that the student doesn't need to inherit.
- **Cross-family friendly.** Even if student and teacher have different bases, the delta abstracts the training-induced shift, which is more transferable than raw log-probs.
- OPD 2 consistently beats vanilla OPD on Korean and Japanese math with Qwen3, narrowing the English–Korean performance gap.

## Gotchas & tricks

- **Delta signal is noisy where the teacher and its base agree.** Vast majority of tokens fall in this regime; the informative signal is concentrated on a small fraction of tokens — the ones post-training actually changed.
- **Requires access to the teacher's base.** Not always available (frontier closed models). Open-weight teachers with released base checkpoints are the easy case.
- **Language drift persists** unless the training data is multilingual — the delta isolates the *shift*, but if the teacher's shift itself is English-biased, so is the student's.
- **Amplifies teacher errors.** Any bad post-training moves in the teacher (over-refusal, verbosity) get transferred with the same fidelity as good ones. Distill from teachers you trust.
- **Not a distillation-from-RL substitute** — it's still supervised: the outcome reward doesn't appear anywhere. For domains where RL discovers behavior the teacher lacks, RL beats OPD 2.

## Sources

- Paper: *On-Policy Delta Distillation for Multilingual Math Reasoning* — 2026 — [arXiv:2608.05802](https://arxiv.org/abs/2608.05802). Introduces OPD 2 for multilingual math with Qwen3.
- Related: On-policy distillation ([on-policy-distillation.md](./on-policy-distillation.md)) — the base method OPD 2 modifies.
