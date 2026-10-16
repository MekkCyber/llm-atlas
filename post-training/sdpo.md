# SDPO — Self-DPO with Charter-Conditioned Judge
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Alignment pass used in Apertus 1.5: a self-play variant of DPO where **preference pairs are judged against an explicit written behavioral charter**. The judge sees the charter in context and scores two candidate responses against it. The policy is trained with DPO on the resulting pairs. Reported win rate vs the plain-DPO checkpoint: **73.2%** (vs 67.4% for self-reflection feedback) and **+2.4%** overall uplift on the 70B development suite.

**Prereqs:** [dpo.md](./dpo.md), [_rewards.md](./_rewards.md)
**Related:** [../safety/README.md](../safety/README.md) · [cot-reward-model.md](./cot-reward-model.md) · [_rl.md](./_rl.md) · [../case-studies/apertus-1-5.md](../case-studies/apertus-1-5.md)

---

## What it is

Standard DPO pits model-generated candidates against each other with a reward model or an LLM judge as the preference oracle. The judge is unconditioned — it decides based on its own prior over "good responses."

SDPO adds an explicit, written **charter**: a document describing the behavior the deployment should exhibit (tone, refusal policy, formatting defaults, safety posture, scope). The judge reads the charter in context and prefers the candidate that best fits the charter. The policy's DPO update aligns it to the charter without ever training the policy itself on the charter text.

## How it works

### The charter

A written document — hundreds to a few thousand tokens — describing the target behavior. Includes positive examples (what the deployment should do), negative examples (what it should avoid), and tie-breakers. The charter is a form of "constitution" (in Anthropic's sense) but used as judge context, not as the training objective directly.

### Pair generation

For each prompt in the alignment dataset, sample $K$ candidate responses from the current policy (self-play; same model).

### Charter-conditioned judging

A judge LLM reads: `(charter, prompt, candidate_A, candidate_B)` and emits a preference. The judge's output is a MaxMin-style pairwise preference (or score) grounded in the charter. In Apertus 1.5, the judge is Qwen3.6-27B.

### DPO update

Collect preference pairs $(x, y_w, y_l)$ where $y_w$ is the judge's preferred response. Standard DPO loss:

$$L_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right].$$

Only the policy sees the DPO gradient; the charter stays with the judge.

## Why it matters

- **Alignment criterion is externalized.** Instead of baking policy preferences into an RM's weights, they live in a readable document. Easier to audit, update, debate.
- **Separates policy capability from alignment target.** The policy learns to produce responses the charter likes; the charter can be revised and the alignment pass re-run without retraining the base.
- **Measured gain:** 73.2% win rate vs the plain-DPO checkpoint on Apertus 1.5 8B, +2.4% on 70B development suite.
- **Compatible with regulatory artifacts.** A charter is a human-readable spec; attaching the alignment pass to it makes compliance review much easier than "we trained an RM on preferences."

## Gotchas & tricks

- **Judge quality bounds alignment quality.** The charter-conditioned judge needs to actually follow the charter. A weak judge collapses SDPO to arbitrary preferences.
- **Charter length vs judge context.** Long charters eat judge context budget; summarized / sectioned charters work better than monolithic documents.
- **Self-play collapse.** As the policy improves, candidate pairs become harder to distinguish, and the DPO signal weakens. Multi-generation refresh (regenerate candidates mid-training) helps.
- **Not a safety mechanism alone.** The charter is only as safe as its author. Red-team charters by trying to find prompts where it would prefer the wrong candidate.
- **Compose with earlier RL.** In Apertus 1.5, SDPO runs after RLVR + DPO; it is a last alignment pass, not a replacement.
- **8B vs 70B.** Apertus 1.5 applies SDPO only to 8B; the 70B ships its plain-DPO checkpoint. The paper doesn't fully quantify the SDPO gap at 70B.

## Sources

- Paper: *Apertus v1.5 Technical Report* — 2026 — [arXiv:2610.15542](https://arxiv.org/abs/2610.15542). Introduces SDPO as the final alignment pass; judge is Qwen3.6-27B with MaxMin preference pairs.
- Paper: *Direct Preference Optimization* — Rafailov et al., 2023 — the DPO loss SDPO reuses. See [dpo.md](./dpo.md).
