# Cross-model prefill-to-autoregressive transfer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Split inference across two different models: a *large* model runs prefill (understands the prompt), a *small* model runs decode (generates the response), connected by a trainable linear bridge over the prefill KV. The gains come from paying prefill FLOPs only once and running the long autoregressive loop on the cheaper model. PACT / PLatour (2611.06976) is the open-source formulation; the same paper also introduces PACT-Check, a diagnostic for whether the gain actually comes from information transfer or just from the small model's own capacity.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** none

---

## What it is

A **disaggregated two-model serving pattern**: prefill is offloaded to a stronger model, autoregressive decode runs on a weaker (cheaper) model, and a small learned module bridges the representation gap. Related in spirit to prefill/decode disaggregation (same model, two phases) and to speculative decoding (small drafter + big verifier), but orthogonal: here the *big* model runs once, the *small* model runs the long loop, and the bridge is trained to transmit useful signal across the two different architectures.

## How it works

Setup: sender $S$ (big), receiver $R$ (small), bridge $B_\theta$.

1. **Sender prefill.** Run $S$ on the prompt, extract its final-layer KV cache $K_S, V_S$.
2. **Bridge.** Project the sender's KV into the receiver's embedding space via **zero-initialized linear bridges** $B_\theta$: $K_R^{(0)} = B_K \cdot K_S$, $V_R^{(0)} = B_V \cdot V_S$. Zero-init means training starts from "receiver ignores sender," which is numerically safe.
3. **Receiver decode.** $R$ autoregressively decodes, with the bridged KV occupying the prefill slots of its context.
4. **Train.** Only $B_\theta$ is trained, with a task-reward objective (e.g. preference loss, answer correctness) on an outer loop.

**PACT-Check** is the paper's methodological contribution: replace the sender's prefill with noise of matched statistics and re-run; if the task reward doesn't drop, the earlier "gain" was not driven by sender-to-receiver information transfer at all.

## Why it matters

- **Serving economics.** Pay a big model's cost once per request; amortize over the long decode on the small model.
- **Composability.** Different model families can be paired — the bridge is the only trained component.
- **Rigor.** PACT-Check retrofits a null baseline to a family of methods (PACT, Mostik) whose published gains were partially driven by artifacts of the small model, not by useful transfer.

Reported results: PLatour matches or beats open-source baselines and Mostik's reported numbers with a smaller sender; part of prior PACT gains fail PACT-Check.

## Gotchas & tricks

- Zero-init bridges are important: with random init the receiver's decode is garbage at step 0 and gradients explode.
- The sender and receiver must share **tokenization** — otherwise the KV offsets don't correspond to the same prompt positions.
- Always run PACT-Check for new PACT variants before claiming gains; replace the sender's prefill with matched-statistics noise and compare.
- Latency win depends on request mix — long-prompt / short-generation workloads benefit; the reverse does not.

## Sources

- Paper: On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer — Lu et al., 2026 — [arXiv:2611.06976](https://arxiv.org/abs/2611.06976)
