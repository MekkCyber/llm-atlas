# Case Study: TeleChat → Xing 4

*A 285B-total / ~17B-active MoE (plus a 29B-A4B open-weight sibling) pretrained entirely on Ascend 910C NPUs. The headline is **not** the final weights — it's that a DeepSeek-V3-shaped stack of co-designed systems and algorithmic innovations can run end-to-end on domestic Chinese silicon.*

**Related concepts:** [mla](../architectures/mla.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [mtp](../pre-training/mtp.md) · [dspark](../inference/dspark.md) · [hsdp](../pre-training/hsdp.md) · [owner-based-muon](../pre-training/owner-based-muon.md) · [dualpipe](../systems/dualpipe.md)

---

## What this is

Xing 4, released November 2026 by TeleAI (China Telecom). Decoder-only sparse MoE:

- **Xing4.0-29B-A4B** — open-sourced. 29B total / 4B active. 40 layers. 64 routed + 1 shared expert, top-4. Native 256K context. Uses **mHC** (manifold-constrained hyper-connections).
- **Xing 4 285B** — ~285B total / ~17B active. 45 layers, hidden 4096. 256 routed + 1 shared experts, top-8. First 3 layers dense. Contexts up to 1M. Pretrained on 8,192 Ascend 910C cards.
- A trillion-parameter member is in training.

What makes it a tech-report-grade release is the breadth: **MLA + DSA + IndexShare** for sparse long-context attention; a **wider-and-shallower MoE** shape tuned to Ascend Cube units; **HSDP** that replaces pipeline parallelism with hybrid sharding on SuperPoD fabric; **owner-based Muon** that shards the Newton–Schulz step; **DSpark**, a block-wise speculative decoder; and **TeleGym**, an asynchronous agent-RL post-training stack. Any one of these alone would be a paper.

---

## Architecture at a glance

```
Xing 4 285B
  45 transformer layers
    ├─ first 3:   attention (MLA + DSA + IndexShare) + dense FFN
    └─ layers 4–45: attention + MoE FFN
         256 routed experts + 1 shared, top-8 routed + shared per token
         expert intermediate dim = 2048  (wider than DeepSeek-V3's 1536)
         fewer layers, more experts per token  → "wider-and-shallower"

  d_model   = 4096
  heads     = standard MLA shape (per-head latent + RoPE slice as in DeepSeek)
  MTP head  retained for training, NOT used at inference (replaced by DSpark)
```

**Why wider-shallower**: Ascend 910C Cube units reward more matrix work per layer. 45 layers × wider MoE (2048) × top-8 beat 60 layers × 1536 × top-6 by ~10% training throughput at matched active params.

**DSA + IndexShare.** DSA (DeepSeek Sparse Attention) adds a lightning indexer that selects top-K historical KV entries per query, so attention reads only a sparse subset of the context — crucial at 1M tokens. **IndexShare** is Xing4's addition: it reuses the sparse indices across layers rather than re-computing per-layer, giving another ~15% end-to-end from avoided indexer work. Simulated attention gives 4.68× prefill and 1.27× decode throughput vs. MLA.

**mHC (29B only).** Manifold-constrained hyper-connections in the 29B residual stream: `n` parallel residual streams with a bounded read/write mixing matrix, giving extra capacity without new sublayer parameters. Not used in the 285B model. The sibling `kg-update/2026-11-18` PR adds a standalone `architectures/hyper-connections.md` page covering mHC and TPHC together.

---

## Training infrastructure

### Parallelism — no pipeline

```
HSDP: FSDP within SuperPoD, replication across SuperPoDs
tensor parallelism:   TP=4
expert parallelism:   EP=64
context parallelism:  CP=32
pipeline parallelism: NONE
```

**HSDP** — hybrid sharded data parallelism — replaces pipeline. Within a SuperPoD, parameters are FSDP-sharded; across SuperPoDs, they are replicated. Per-layer AllGather/ReduceScatter stay intra-SuperPoD (fast); only coarse gradient reduction crosses SuperPoDs. PyTorch-FSDP's flattened-buffer copy-in/out is removed with zero-copy per-parameter collectives. During layer *i*'s backward, the schedule overlaps layer *i+1*'s ReduceScatter, layer *i−1*'s AllGather, and inter-SuperPoD AllReduce.

Why it wins: the 1F1B pipeline bubble is 3/11 (~27%) at p=4, m=8. Even interleaved pipeline (vpp=2) only cut step time from 36.4 s to 34.2 s. HSDP reached **31.8 s — 1.14× faster than 1F1B at the same cluster**. See [hsdp](../pre-training/hsdp.md).

### Owner-based Muon

Muon's Newton–Schulz orthogonalization step needs the full matrix. In standard sharded Muon, every rank duplicates the NS computation on the gathered matrix. Xing4 assigns **one owner rank per matrix** that does NS; non-owners P2P-send their shards, owner runs NS, scatters the result back. Parameters and momentum stay sharded; only the NS compute moves. Step time: **31.8 s → 28.9 s (1.10×)**. See [owner-based-muon](../pre-training/owner-based-muon.md).

### Fine-grained selective recomputation

Operators chosen with a knapsack formulation over measured time/memory. Boundary activations released and restored on demand. Reports another 6.9% step-time reduction over the HSDP+Muon baseline.

### Indexer backward pass

Layer-wise indexer backward with Q-block head–tail pairing, document-aware KV compaction, cost-based DP balancing. Another 15.2% step-time reduction.

### Hardware

Ascend 910C NPUs with MindSpore HyperParallel. The 285B pretrain used **8,192 cards** from a production cluster; profiling used 1,024 NPUs; inference simulation used 32 cards. No training cost, duration, or GPU-hour number is reported.

---

## Pre-training

- **Data.** Pretraining corpus mixture, token count, and language balance **not disclosed**.
- **Objectives.** Standard MoE pretraining + aux-loss-free balancing + MTP auxiliary head (λ not stated).
- **Context extension.** Up to 1M tokens for the 285B model, 256K native for the 29B. Mechanics not detailed in the abstract.
- **Optimizer.** Muon (owner-based variant).

---

## Post-training

**TeleGym** — an agent-RL harness in the Slime + SGLang mold:

- Heterogeneous agent harnesses run in sandboxes.
- A server converts interactions into token-level trajectories with sampling-time log-probs and policy versions.
- Generation and training run **asynchronously on separate device pools**.
- **Partial rollout at policy sync**: generation pauses at a safe trajectory boundary, unfinished trajectories resume under the new policy. Optional stale-data filtering discards whole query groups.
- Multi-turn episodes count as one attempt for group-relative rewards — compatible with GRPO-family losses.

**RL algorithm, reward design, and hyperparameters are not disclosed**. The partial-rollout pattern is the same shape as DeepSeek's and Kimi K1.5's (see [partial-rollouts](../systems/partial-rollouts.md)).

---

## Inference: DSpark

**DSpark** is a block-wise speculative decoder that replaces MTP-at-inference. Five-layer DFlash drafter + **Markov head** (dependency correction between adjacent drafts) + **confidence head** (per-position acceptance estimator). Proposes **7 positions in parallel**. See [dspark](../inference/dspark.md).

| Benchmark | Base tok/s | MTP3 tok/s | DSpark tok/s | DSpark × base |
|---|---|---|---|---|
| GSM8K | 35.4 | 63.7 | **115.8** | 3.27× |
| HumanEval | 35.7 | 64.1 | **134.7** | 3.77× |

Across six math/code benchmarks: **2.67×–3.77× over base, 1.48–2.10× over MTP3**. Acceptance rate not reported numerically.

---

## Evaluation snapshot

The body **does not report general-capability benchmark numbers** (MMLU, GSM8K accuracy, AIME, HumanEval, etc.) for the final models. All quantitative results are systems metrics: training step time, inference throughput, simulated attention throughput, etc. This is unusual for a tech report and worth flagging — Xing4 is validated on systems claims, not on head-to-head model-quality claims with DeepSeek-V3 / Qwen3 / Kimi.

---

## Key takeaways

1. **Pipeline parallelism is replaceable on dense enough fabric.** HSDP + per-parameter zero-copy collectives + comm overlap gets you the throughput of DualPipe with none of its scheduling fragility. On SuperPoD-class interconnects (and plausibly on NVLink-rich H100/B200 clusters), PP is no longer load-bearing.

2. **Muon can be sharded by matrix ownership.** The NS step duplication across ranks is an artifact of the obvious implementation, not a necessity. Owner-based Muon gives 10% training-step time at frontier scale for ~zero quality risk.

3. **Block-wise speculative decoding is the evolved form of MTP-at-inference.** MTP-3 is a stepping stone; a short drafter + dependency head + confidence estimator hits 2–3× higher throughput with the same self-consistency guarantee.

4. **Wider-shallower MoE is hardware-conditional.** On Ascend Cube units, 45 layers × 2048 expert × top-8 beat 60 × 1536 × top-6 by ~10%. On H100/B200 the balance shifts; the lesson is to co-design MoE shape with the matmul primitive.

5. **IndexShare is a cheap compounding win.** Reusing DSA indices across layers is a one-page idea with a ~15% throughput effect at 1M context. Expect every long-context sparse-attention stack to adopt it.

6. **Reporting style is systems-first.** No accuracy benchmarks in the body is a conscious choice — this is a tech report about the **training and serving pipeline**, with weights and model-quality claims deferred to community evals. Compare to OLMo 2 (full transparency across both) or DeepSeek-V3 (both axes).

---

## What's still opaque

- **No general-capability benchmarks** for either model. Model quality has to be assessed by independent evaluators on the open 29B.
- **No training cost or duration.** The 8,192-card 910C pretrain carries no wall-clock or kcard-hour figure.
- **Pretraining data details** — token count, mixture, language balance, decontamination — are entirely absent.
- **RL algorithm and reward model** for TeleGym post-training are not specified beyond "GRPO-compatible."
- **No tokenizer or vocab specifics.**
- **FP8 or other precision choices** for the 910C pretrain are not disclosed; the 910C's native BF16 story is well-known but the Xing4 report doesn't commit either way.

---

*Pairs well with:* the [DeepSeek-V3 case study](deepseek-v3.md) for direct comparison — same tech-report shape, same breadth of systems+algorithm co-design, but V3 ran on H800 and reported full evals, while Xing4 ran on Ascend 910C and reported systems metrics only. Different hardware constraints yield different innovation emphases (DualPipe vs. HSDP, MTP vs. DSpark, standard Muon vs. owner-based).
