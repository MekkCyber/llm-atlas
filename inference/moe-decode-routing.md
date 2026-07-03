# MoE Decode Routing (Expert-Locality-Aware Scheduling)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In prefill/decode-disaggregated MoE serving, two decode workers with the same measured *load* can serve at very different latencies because each decode step must load the weights of **every distinct expert its batch activates**. Route each request to the decode worker whose hot cache best matches its expected expert set. ELDR (2026) does this offline: build an **expert signature** from each request's prefill activations, K-means-cluster the signatures, and use locality-band routing at decode time.

**Prereqs:** [pd-disaggregation](pd-disaggregation.md), [../architectures/_moe.md](../architectures/_moe.md)
**Related:** [../architectures/deepseek-moe.md](../architectures/deepseek-moe.md), [../architectures/aux-loss-free-balancing.md](../architectures/aux-loss-free-balancing.md)

---

## What it is

Standard LLM serving schedulers optimise for *load balance*: keep worker queues equal. For MoE this is under-specified — two workers with the same queue depth can spend very different wall clock per token depending on **which experts** their in-flight batches touch. Fine-grained MoE (256+ experts, top-8 routing) makes each decode step gate on the union of experts across the current batch; low-locality batches load many more experts per step than high-locality ones.

MoE decode routing is the class of schedulers that **treats expert overlap as a first-class scheduling criterion**, not just queue length.

## How it works

The clean lever is: prefill already computes routing decisions for the whole prompt. Aggregate those into a per-request **expert signature** — for each MoE layer, the set (or histogram) of experts the prompt activated. This signature is a strong predictor of which experts the decode phase will need.

ELDR structures it as:

- **Offline clustering** — collect prefill signatures from historical requests, run balanced K-means over the signature space to find $K$ **locality clusters** (each cluster is a subspace of expert-space that requests tend to share).
- **Online routing (locality band)** — at decode-scheduling time, each cluster is assigned a **band** of decode workers. New requests are routed to their cluster's band, so within-band workers see requests whose signatures overlap and whose expert weights are already hot.
- **Coherence with prefix caching** — the scheduler keeps signature-based routing consistent with prefix-cache stickiness (identical prefixes hit the same worker), so both caches (KV prefix + MoE expert weights) win simultaneously.

At inference time no extra model calls are needed — the signature is a byproduct of prefill, and cluster lookup is a hash.

## Why it matters

- **Median TPOT ↓ 5.9–13.9%** vs. load-balancing baselines across three MoE models (ELDR paper), with gains scaling in expert count.
- **Fine-grained MoE is the frontier default** (DeepSeekMoE, DeepSeek-V3, MoE frontier open models). Locality-oblivious schedulers waste more of the expert-cache advantage the more experts you have.
- **Uses signals that already exist.** No new model calls, no retraining — the scheduler consumes information the serving stack already computes.

## Gotchas & tricks

- Signature drift: workloads change hourly (traffic-of-day, feature launches). Re-cluster on rolling windows, not once.
- Cluster band sizing is the real tuning knob — too narrow bands re-introduce load imbalance; too wide bands re-introduce cache thrash.
- Under low-locality workloads (many domains mixed), the signature space fragments and gains shrink; ELDR-style routing is most valuable when the workload has genuine topical clustering.
- Composes with (not replaces) queue-length balancing — locality routing picks a band; within a band, standard load balancing chooses the worker.
- For MoE + PD-disaggregation, the prefill stage is where the signature is cheapest to compute; a decode-only cluster loses the signal.

## Sources

- Paper: *ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving* — Cho, Xiong, Yang, Kwon, Cheng, 2026 — [arXiv:2607.00466](https://arxiv.org/abs/2607.00466).
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — fine-grained MoE regime where locality wins are largest.
