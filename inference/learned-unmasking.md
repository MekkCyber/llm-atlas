# Learned Unmasking for Diffusion Language Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Masked diffusion language models decode tokens in parallel; at each step a policy decides **which** predictions to commit. A good unmasking policy is the single biggest lever on the accuracy–NFE (number of function evaluations) frontier. A controlled study finds that observing **confidence + hidden states** and training the policy on **trajectory-contraction targets** gives the best frontier. An 856K-parameter policy on top of LLaDA-8B-Instruct wins GSM8K throughput and largely transfers to Dream-7B-Instruct (NTU Singapore, 2026).

**Prereqs:** [README.md](README.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

Masked diffusion LMs iteratively unmask tokens from an all-masked sequence. Decoding speed is governed by how aggressively the per-step unmasking policy commits predictions. A good policy commits confident, correct tokens (high throughput, high accuracy) and holds uncertain ones (avoids propagating errors). The paper is a controlled study of the design choices that go into training such a policy.

## How it works

Four design axes studied:

1. **Teacher label.** What counts as "committable" at each position? Options: predictions that match the final output (permissive), predictions above a confidence threshold (strict), or trajectory-contraction (predictions that cause the DLM to need fewer steps to converge). The last gives the best accuracy–NFE frontier at low NFE; "matches final output" is too permissive at low NFE and hurts accuracy.
2. **Imitation states.** Which trajectory states does the policy learn on? Early-step states, late-step states, or a mix. Mixing is best.
3. **Policy inputs.** What does the policy observe at each position — confidence alone, hidden states alone, or both? **Both** is strictly better; hidden states alone are weak but complement confidence; confidence alone misses structural information.
4. **NFE budget.** How much must NFE drop to amortise policy overhead? The paper fixes a budget and measures accuracy at matched NFE; the policy overhead is small enough that modest NFE reduction already yields net latency wins.

Trained policy: **856K parameters** (orders of magnitude smaller than the 8B base), trained on trajectory-contraction targets with confidence + hidden-states input.

## Why it matters

- **Diffusion LMs promise parallel decoding; the unmasking policy is where that promise lives or dies.** This paper gives a reproducible recipe that *actually* yields net latency wins.
- **Transferability.** The 856K-param policy trained on LLaDA-8B largely transfers to Dream-7B without retraining. Suggests the design choices generalise across diffusion-LM backbones.
- **Tiny policy, big lever.** Policy cost is negligible next to the base. The accuracy–NFE improvement is almost pure win.
- **Methodological contribution.** The four-axis decomposition is a template for anyone studying diffusion-LM decoding policies — not just this specific architecture.

## Gotchas & tricks

- **"Matches final output" is the trap.** Labelling every eventual-match as committable sounds right but is too permissive at low NFE — the policy commits unconfident tokens that happen to be correct, which doesn't train the right generalisation.
- **Hidden states need to be from the right layer.** The paper reports best-layer selection as a tunable choice; using the final-layer representation alone is worse than mixing mid-layer features.
- **NFE savings depend on the base's confidence distribution.** On well-calibrated bases, confidence alone works; on miscalibrated bases, hidden states carry the real signal.
- **Doesn't accelerate every task equally.** GSM8K throughput gains are the paper's headline; open-ended generation sees different NFE floors.

## Sources

- Paper: *What Makes a Good Unmasking Policy? A Controlled Study of Learned Unmasking in Diffusion Language Models* — Sakurai, Xu, Yoon, NTU Singapore, 2026 — [arXiv:2610.31576](https://arxiv.org/abs/2610.31576).
