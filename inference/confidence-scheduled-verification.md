# Confidence-scheduled verification
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In speculative decoding, verification length has traditionally been a fixed hyperparameter. Confidence-scheduled verification makes it a **per-request, load-aware online decision**: use draft-side confidences to estimate prefix-survival probabilities, combine with the serving engine's current throughput profile, and pick the verification length $k^*$ that maximizes net accepted tokens under current load. Introduced in DSpark; shifts the Pareto frontier of latency vs throughput in DeepSeek-V4 serving.

**Prereqs:** [dspark.md](./dspark.md)
**Related:** [_speculative-decoding.md](./_speculative-decoding.md), [../pre-training/mtp.md](../pre-training/mtp.md)

---

## What it is

Standard speculative decoding fixes the draft length $K$ (and thus verify length) globally: e.g., "always draft 4 tokens, always verify 4." Under load variance this is suboptimal on both sides:

- **Under low load** with lots of spare batch capacity, longer verifies are cheap and even tail-accepted tokens are net-positive.
- **Under high load** with contended batches, verifying long low-confidence drafts wastes capacity on tokens that will be rejected — hurting throughput for *other* requests.

Confidence-scheduled verification treats $k^*$ as an online decision per request per step.

## How it works

Two inputs, one output:

**Input 1 — prefix survival probability.** For each candidate position $i \in [1..K]$ in the draft, estimate $P_i$ = probability that positions $1..i$ all get accepted. Computed from draft-side per-token confidences (calibrated during training). $P_i$ decays as $i$ grows — the expected number of accepted tokens conditional on verifying up to $i$ is $\sum_{j \le i} P_j$.

**Input 2 — engine throughput profile.** The serving system knows its current cost function: how does an extra verified token affect batch capacity, memory, and other requests at the current concurrency level? Call this $C_i$ — the marginal cost of extending the verify to position $i$.

**Choice.** Pick the $k^*$ that maximizes net benefit: continue extending the verify while $\Delta P_i > \lambda \cdot \Delta C_i$ for a load-dependent $\lambda$; stop when the marginal accepted token is no longer worth its cost.

The engine throughput profile can be as simple as a lookup table indexed by current concurrency, or fit online.

## Why it matters

- **Directly enables new latency tiers.** DSpark reports that fixed-$K$ scheduling made certain per-user tokens/s under concurrency SLOs unreachable; confidence-scheduling opens them up by conserving batch capacity when it's tight.
- **Turns speculative decoding from a fixed-throughput trick into a Pareto-frontier operator.** Same drafter and verifier — different scheduling — measurable production gains.
- **Composable with any drafter.** Not tied to semi-AR: MTP, Medusa, EAGLE all benefit from the same online verify-length choice.
- **Load-aware inference is generalizable.** Same principle applies to KV-cache retention, prefetching depth, batch composition — the "make it adaptive to current load" pattern.

## Gotchas & tricks

- **Calibration is essential.** If drafter confidences are miscalibrated (over-confident), $P_i$ estimates are wrong and $k^*$ chases the wrong optimum. Calibration should be part of drafter training.
- **The throughput profile drifts with request mix.** Different prompt lengths, different sequence lengths, cache pressure — all shift $C_i$. A static profile is a starting point; production systems will need to update it online.
- **Adaptive verify requires kernel support.** The engine has to accept variable-length verify batches, not just fixed $K$. Custom masking or padding is needed.
- **Introduces a per-request scheduling decision.** Not free — but cheap enough to run per step (a few multiplications) if the profile is prefetched.
- **Different from adaptive draft length.** Draft length is a drafter choice; verify length is a target-side choice. DSpark tunes both, but they're distinct axes.

## Sources

- Paper: *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation* — Yu, Shao, Li, et al., DeepSeek-AI / Peking University, 2026 — [arXiv:2607.05147](https://arxiv.org/abs/2607.05147).
