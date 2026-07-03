# Test-Time Scaling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Sample $N$ responses to one prompt, then pick one — the "just keep sampling" recipe behind pass@$N$ / best-of-$N$ / self-consistency. **Coverage** (probability a correct answer appears in the $N$ draws) climbs with $N$, but **selection** (choosing which draw to return, not knowing which is right) is bounded by two hard ceilings — the **modal ceiling** (majority vote stabilises within a few dozen draws) and the **correlation ceiling** (benchmark score correlation saturates earlier still). Extra samples past that only reinforce confident mistakes.

**Prereqs:** [../rejection-sampling](../rejection-sampling.md), [../_rl](../_rl.md)
**Related:** [orm](orm.md), [prm](prm.md), [long-cot-rl](long-cot-rl.md)

---

## What it is

Test-time scaling is the family of inference-time techniques where compute is spent on **multiple samples** rather than a longer single response:

- **pass@$N$** — the eval metric: is there at least one correct answer in $N$ draws?
- **best-of-$N$ (BoN)** — pick the highest-scoring draw under a verifier / RM. See [orm](orm.md).
- **self-consistency** — pick the modal answer (majority vote over final answers).
- **reward-model reranking** — score each draw with an RM, argmax.

All four rest on a *coverage → selection* pipeline: first produce candidates that cover the correct answer, then select one. The literature has focused on coverage; the modal-ceiling analysis shows selection is the actual bottleneck.

## How it works

Two independent limits interact.

### Coverage vs. selection

Let $p$ be the per-sample probability of a correct answer. Coverage under $N$ i.i.d. draws is $1 - (1-p)^N$, growing monotonically toward 1. But **selection** — pick one draw to return — is capped:

- **Modal selection (majority vote):** returns the most common answer. As $N$ grows the modal answer converges to $\arg\max_a P(\text{draw} = a)$, which need not be the correct answer for hard problems. Once it converges, more draws only sharpen the wrong choice.
- **Verifier / RM selection:** returns the highest-scored draw. Bounded by the verifier's precision on the drawn candidate set — spurious high-scoring draws re-cap the win.

The gap between climbing coverage and stalled selection is the **identifiability gap**: answers the model can produce but can't pick.

### Modal ceiling

For modal selection, the majority vote settles within a few dozen draws — $N \approx 20\text{--}64$ typically — after which the returned answer stops changing. The modal-ceiling result names this cutoff and formalises it as an **effective number of samples** $N_{\text{eff}}$ that any sampling run reveals from its own draws: $N_{\text{eff}}$ is where the vote has stabilised, and further draws are wasted compute.

### Correlation ceiling

For benchmark scoring, aggregated correlation with true accuracy saturates sooner still. Even before the vote stabilises per-prompt, the *ranking* of models by their test-time-scaled scores stops moving — so bigger $N$ neither reranks models nor sharpens the eval.

## Why it matters

- **Compute wasted at the tail.** Doubling $N$ past $N_{\text{eff}}$ doubles inference cost with no accuracy gain — sometimes a loss, if you're overrepresenting a confident wrong mode.
- **Reframes the RL story.** Since selection is the bottleneck, the highest-leverage moves are training the **verifier / RM** (raise the selection ceiling) or training the **generator** to make more of its coverage identifiable (raise $p$ correlated with confidence). Both are what RLVR / ORM / PRM are already doing.
- **Prompt-difficulty implication.** For prompts where the modal-ceiling answer is wrong, test-time scaling *hurts*: extra draws add cost and make the wrong answer surer. Difficulty-aware early stopping recovers the loss.

## Gotchas & tricks

- The modal ceiling is per-prompt: hard prompts saturate at low $N$ with the wrong answer, easy prompts saturate quickly with the right one. Aggregate $N_{\text{eff}}$ can hide this.
- Verifier-based selection can *exceed* the modal ceiling — that's what ORMs are for. But the verifier's own error rate re-caps the win; a bad verifier makes BoN converge to the same modal answer as majority vote.
- Diversity tricks (higher temperature, top-p, prompt variants) raise coverage without necessarily raising selection — the gap can widen, not narrow.
- Empirically the modal ceiling holds even for very strong reasoners; scaling test-time compute much beyond a few dozen draws should be justified by an identifiability-closing mechanism (verifier, self-correction, tool use), not by "just more samples."

## Sources

- Paper: *When More Sampling Hurts: The Modal Ceiling and Correlation Ceiling of Test-Time Scaling* — Bay & Yearick, 2026 — [arXiv:2606.28661](https://arxiv.org/abs/2606.28661).
- Paper: *Self-Consistency Improves Chain of Thought Reasoning in Language Models* — Wang et al., 2022 — majority-vote origin.
- Paper: *Let's Verify Step by Step* — OpenAI, 2023 — verifier-based BoN as the identifiability-gap closer.
