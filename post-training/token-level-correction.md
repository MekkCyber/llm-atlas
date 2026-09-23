# Token-Level Correction (for On-Policy Alignment Data)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An annotation workflow where a human, reading a model response, finds the *first wrong token*, picks a replacement from the model's candidate tokens (or types a short edit), and the model regenerates from that corrected prefix. Iterate until the whole answer is good. Since most of the final response is still model-generated, the resulting text stays on-policy — it's suitable as SFT data for the same model — and each correction yields a per-position (wrong-token, right-token) supervision signal that doubles as paired positive/negative preference data.

**Prereqs:** [_post-training.md](_post-training.md), [rejection-sampling.md](rejection-sampling.md), [rlvr.md](rlvr.md)
**Related:** [dpo.md](dpo.md) · [../agents/README.md](../agents/README.md)

---

## What it is

Traditional alignment data is either **human-written** (off-policy, drifts the model) or **model-drafted then human-edited** (post-hoc rewriting). Both change the response distribution enough that the resulting SFT data pushes the model away from its own sampling distribution — a well-known cause of degraded generation quality.

**Token-level correction** flips the workflow: instead of editing a completed response, the annotator *steers* the response mid-generation, one token at a time:

1. Model streams tokens.
2. Annotator spots the first inappropriate token at position $p$.
3. Annotator either **picks** a substitute from the top-K candidates shown at $p$, or **types** a short free-form replacement.
4. System truncates the response after $p$ and resumes generation from the new prefix.
5. Repeat until the response is acceptable.

The final response is a mix: mostly the model's own tokens, with a small number of human-selected corrections at pivotal positions.

## How it works

**On-policy-ness** is preserved because the vast majority of tokens are still sampled from the current model. A correction at token $p$ changes the *prefix* for all subsequent generation, so downstream tokens are also on-policy conditioned on the corrected prefix.

**Token-level supervision** falls out for free:

- Every correction at position $p$ produces a triple $(\text{prefix}_{<p},\; \text{wrong-token},\; \text{right-token})$ — precise position, precise sign, precise gradient direction.
- The (wrong-token, right-token) pair is a natural preference pair for DPO-style training at token granularity.
- The full corrected response is on-policy SFT data — no distribution shift to correct for.

**Trajectory annotation.** The workflow generalizes past chat: connect the annotator's UI to external tools and environment observations, and the same locate-correct-continue loop annotates agent trajectories mid-rollout. Corrections happen at any step, not just at text-token boundaries.

## Why it matters

- **Cheap and dense at once.** Preference annotation is cheap-but-sparse (one label per response); rewriting is dense-but-expensive. Token-level correction gets both — the annotator only edits the tokens that matter.
- **Cuts annotation time ~in half.** A controlled user study reports **-52% median annotation time** vs. manual post-editing.
- **On-policy data for free.** No need to rebuild a distribution-matched preference set after each model update — the annotator's corrections are always drawn against the current model's outputs.
- **Extends to agents.** Because the loop hooks external environments, trajectory-level corrections come out with the same UX.

## Gotchas & tricks

- **The candidate-list UX matters.** Presenting top-K logits at the correction position (rather than only free-form editing) speeds annotation dramatically — most fixes are just picking the second-best token.
- **Corrections change the prefix distribution.** After a substitution, the model resumes generation conditioned on a token *it did not* rank first — so the downstream tokens live in a slightly off-manifold region. Multiple corrections in a row compound this drift; a rule of thumb is to accept the response after $\leq$ 3–5 corrections and start over if more are needed.
- **Position tokens are the label, not just the response.** If you throw away the correction positions after generating the response, you've reduced the data back to plain SFT and left the token-level preference signal on the floor.
- **Not the same as inline RLHF.** The annotator provides positive-and-negative *tokens* for training; the reward model is implicit in the annotator's choices.
- **Requires a controllable serving stack.** The generator has to expose top-K candidates and support "truncate-and-continue" — vLLM/SGLang can do this, but the UI plumbing is a build.

## Sources

- Paper: *onPanda: Efficient Annotation of On-Policy Alignment Data for LLMs and Agents via Token-Level Correction* — Yang et al., StepFun, 2026. [arXiv:2609.24983](https://arxiv.org/abs/2609.24983)
- Dataset: *Panda-CVL* — the token-level-annotated agentic/vision dataset released with the paper.
- Related: DPO, sequence-level rejection sampling, on-policy distillation.
