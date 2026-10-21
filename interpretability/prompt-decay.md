# Prompt Decay (Conditional No-Hair Theorem)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A theoretical and empirical result on how fast a decoder-only transformer forgets its prompt during generation. Feed two different prompts followed by the same continuation; the gap between their hidden states shrinks roughly as a **power law** in the number of generated tokens. Trained models forget faster than untrained models with the same architecture. The paper proves an upper bound on the gap for pre-LN transformers under a non-degeneracy condition, and shows via a solvable linear model that the bound is tight. Mechanism: attention on the prompt is diluted as context grows, *plus* a "relay" effect where continuation tokens pass along prompt information they've already absorbed.

**Prereqs:** [attention](../fundamentals/attention.md), [_normalization](../architectures/_normalization.md)
**Related:** [rope](../fundamentals/rope.md), [dca](../fundamentals/dca.md)

---

## What it is

Practitioners routinely observe "lost in the middle" and "the model stopped following the system prompt". This paper formalises the underlying dynamic: *the information the hidden state retains about a prompt decays as the model generates more tokens*, and the decay rate has a shape and a theoretical floor.

The authors measure prompt retention by the hidden-state gap: for two different prompts $P_1, P_2$ followed by the *same* continuation $C$, how far apart are the hidden states at position $|P| + t$? If the model perfectly remembers the prompt, the gap is large (two different prompts yield different states); if the model has forgotten, the gap shrinks toward zero.

## How it works

**Empirical finding.** The hidden-state gap scales roughly as $\|h_t(P_1) - h_t(P_2)\| \sim t^{-\alpha}$ for some $\alpha > 0$ that depends on the architecture and training. Trained models have *larger* $\alpha$ (faster forgetting) than randomly-initialised models with the same architecture.

**Upper bound (pre-LN transformers).** Under a non-degeneracy condition on the attention pattern (loosely: the softmax doesn't collapse to a single token), the gap at step $t$ is upper-bounded by $C \cdot t^{-\beta}$ for an explicit $\beta$ depending on layer count and residual-stream scaling. The bound is tight in a solvable linear-attention model.

**Two mechanisms.**
- **Dilution.** As the context grows, softmax attention mass shifts to the new tokens; any individual prompt token's weight decays as $O(1/t)$ by softmax saturation.
- **Relay effect.** Continuation tokens absorb prompt information in their own hidden states and then are themselves attended to by later tokens. Prompt info reaches later positions through this relay, but each relay step averages out the prompt-specific signal.

**Counter-move: logit bias on prompt attention.** Adding a positive bias to attention logits that target prompt positions slows (or reverses, if large enough) the decay. A mild bias improves adherence to system instructions at a small perplexity cost.

## Why it matters

- **Theoretical floor on prompt adherence.** The decay is not a bug in post-training — it is a geometric consequence of softmax attention over a growing context. Chat templates and "I will restate the system prompt" tricks are papering over this.
- **Explains why trained models forget faster than fresh models.** Training sharpens attention toward recent tokens, which steepens the decay. A tension between fluency (favour recent tokens) and prompt adherence (preserve prompt dependence).
- **Attention bias is a practical, cheap fix.** The logit bias doesn't require retraining or RLHF; it is applied at inference. Precedent for a growing class of inference-time attention-steering interventions.
- **Interpretability target.** The relay effect gives a concrete mechanism to probe — residual-stream directions that preserve prompt information across many relay steps.

## Gotchas & tricks

- **The bound is for pre-LN.** Post-LN transformers behave differently (residual gain vs normalisation gain ordering). The paper covers only pre-LN; practitioners on post-LN stacks should re-derive before relying on specific $\beta$ values.
- **Non-degeneracy condition can fail.** When attention collapses (very long context with low entropy — e.g. repeated boilerplate), the bound doesn't apply and the gap can decay even faster.
- **Logit bias trades perplexity for adherence.** Too large a bias degrades fluency; the paper's sweet spot is small (single-digit logits).
- **Linear-attention variants are not immune.** The relay-effect part of the mechanism carries over to any architecture that mixes past-token information through a residual stream, including SSM / Mamba. Only the dilution rate changes.
- **Not a replacement for context-length engineering.** Prompt decay compounds with other long-context failure modes (RoPE extrapolation, DCA artefacts); treat them additively, not as a single phenomenon.

## Sources

- Paper: *A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt* — anonymous, 2026 — [arXiv:2610.24133](https://arxiv.org/abs/2610.24133).
- Background: *Lost in the Middle: How Language Models Use Long Contexts* — Liu et al., 2023 — empirical precursor.
