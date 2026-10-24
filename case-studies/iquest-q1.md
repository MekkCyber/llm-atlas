# Case Study: IQuest-Q1

*A 320B-total / 15B-active sparse-MoE agentic foundation model, open-weights, specialised for agentic coding. The release is interesting less for the headline parameter count and more for its three-stage recipe — a stability-first pre-training stack, a fine-grained mid-training that progressively extends context to 512K, and a "multi-scaffold multi-version" post-training that combines SFT, domain-specialised RL, MOPD (multi-teacher on-policy distillation), and model merging across experts and checkpoints.*

**Related concepts:** [_moe.md](../architectures/_moe.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [mid-training](../pre-training/mid-training.md) · [stable-splitnorm](../pre-training/stable-splitnorm.md) · [model-souping](../pre-training/model-souping.md) · [mtp](../pre-training/mtp.md) · [_rl](../post-training/_rl.md) · [rlvr](../post-training/rlvr.md) · [domain-specialized-rl](../post-training/domain-specialized-rl.md) · [mopd](../post-training/mopd.md) · [rejection-sampling](../post-training/rejection-sampling.md) · [deepseek-v3](deepseek-v3.md)

---

## What this is

IQuest-Q1, released 2026 by IQuest. A sparse mixture-of-experts decoder transformer: **320B total parameters, 15B activated per token**, trained from scratch on **29T tokens** (pre-training + mid-training combined). Released as open-source. The design brief: a frontier-class model whose dominant target is **agentic coding** — SWE-bench-Multilingual, Terminal-Bench, agentic CLI workflows — rather than a one-shot chat model.

The report matters because of what it bundles together: **Stable SplitNorm**, **Muon** optimizer, **µP** parameterization, a two-stage pre-training curriculum that preserves a wide stability margin between safe and optimal learning rates, progressive context extension in mid-training up to 512K, and a post-training stack combining SFT, domain-specialised RL, and MOPD across multiple scaffolds, finished with model merging across experts and stages. Any one of these would be a real contribution on its own.

---

## Architecture at a glance

```
88 transformer layers
  ├─ layer 1:   attention + dense FFN (intermediate 12 288)
  └─ layers 2–88: attention + MoE FFN (256 routed, top-8 active)

d_model       = 3 072
heads         = 48 query / 8 key-value, d_head = 128   (GQA)
attention mix = 3 : 1  sliding-window : full attention,
                  sliding-window = 4 096 tokens
MoE per layer = 256 routed experts, 8 active per token
                expert intermediate dim = 1 536  (fine-grained)
                first layer is dense, not MoE

vocab         = 160 000  (untied input / output embeddings)
context       = progressively extended to 32K → 128K → 512K
                during mid-training

+ MTP head: 2 layers at training time;
            at inference, 1 layer is reused recursively for
            self-speculative decoding
```

The architecture is DeepSeek-V3-adjacent (fine-grained MoE, MTP), with a few distinctive choices: a 3:1 **sliding-window to full-attention ratio** at 4 096-token local windows, **GQA** rather than MLA, a dense first layer, and extremely aggressive context extension to **512K** during mid-training.

---

## Training recipe

### Stage 1 — Stable and efficient pre-training

**Core techniques.**

- **Stable SplitNorm** (new; see [stable-splitnorm](../pre-training/stable-splitnorm.md)) — a normalization variant whose purpose is to preserve a wide stability margin between the *safe* and *optimal* learning rates. Lets the recipe pick an LR close to the training-divergence boundary without divergence risk.
- **Muon** for weight matrices; **Adam** for embeddings, the output head, norms, and routers. Adam β1 = 0.9, β2 = 0.95, ε = 1 × 10⁻¹⁵. Weight decay 0.1. **QKNorm clipped at 10**.
- **µP parameterization** to transfer tuned hyperparameters across model widths.

**Two-stage curriculum (within Stage 1).**

- **Stage 1a:** standard mixture.
- **Stage 1b:** code share rises **40 % → 70 %** across four phases, biasing the model toward agentic coding targets before mid-training.

**Learning rate schedule.** Warm up to **6 × 10⁻⁴**; decay to **2 × 10⁻⁴** by end of Stage 1; later decay to **6 × 10⁻⁵** by end of mid-training.

### Stage 2 — Fine-grained mid-training

- Expands the data mixture across **domains and formats**.
- **Progressive context extension** to **32K → 128K → 512K**. The 512K reach is extreme by current frontier-model standards and is bought here with mid-training time rather than being pushed into post-training as a separate "long-context SFT" stage.
- Combined with Stage 1, the pre-training + mid-training budget is **29T tokens** total.

See [mid-training](../pre-training/mid-training.md) for the general pattern; IQuest-Q1 is a strong example of mid-training as the *context-extension* stage rather than the *data-shift* stage alone.

### Stage 3 — Multi-scaffold with multi-version post-training

The paper's name for its post-training stack. Three ingredients, applied across multiple *scaffolds* (different task framings) and multiple *versions* (different checkpoints kept for merging):

1. **SFT** in two sub-stages: **~240B tokens** at peak LR **5 × 10⁻⁵**, then **~50B tokens** at peak LR **1.5 × 10⁻⁵**. The second sub-stage is a lower-LR polish focused on agentic-coding targets.
2. **Domain-specialised RL** — see [domain-specialized-rl](../post-training/domain-specialized-rl.md). Each capability cluster (coding, tool-use, general reasoning) has its own reward routing and roll-out environment, so RL signal doesn't average across capability modes.
3. **MOPD — Multi-teacher On-Policy Distillation** — see [mopd](../post-training/mopd.md). A committee of specialist RL-trained teachers steer a single student *on-policy* during rollouts.

**Model merging.** The final step: merge experts and checkpoints from different stages, with conflict mitigation across parameter updates. See [model-souping](../pre-training/model-souping.md) for the general technique; IQuest-Q1 pushes merging to span not just checkpoints but *expert pools* and *post-training versions*.

---

## Evaluation snapshot

Selected benchmarks from the paper's eval tables. (IQuest-Q1 is a chat/agentic model; comparisons to GPT-/Claude-class systems are available in the full report.)

| Benchmark | IQuest-Q1 |
| --- | --- |
| **Coding** | |
| SWE-bench Multilingual | 80.3 |
| SWE-Bench Pro | 57.5 |
| DeepSWE v1.1 | 64.6 |
| LiveCodeBench v6 | 89.7 |
| Terminal-Bench 2.1 | 83.2 |
| CyberGym | 84.5 |
| PaperBench | 86.0 |
| NL2Repo-Bench | 63.0 |
| ProgramBench | 60.9 |
| **General / reasoning / agentic** | |
| GPQA Diamond | 93.3 |
| IMOAnswerBench-Verified | 93.0 |
| Humanity's Last Exam | 39.2 (55.7 with tools) |
| BrowseComp | 85.5 |
| WideSearch | 81.8 |
| Toolathlon-Verified | 75.0 |
| Claw-Eval | 77.1 |
| JobBench | 55.7 |
| APEX-Agents 1.1 | 50.9 |

Standout: **SWE-bench Multilingual at 80.3** and **LiveCodeBench v6 at 89.7** anchor the "agentic coding" positioning; **GPQA Diamond at 93.3** and **IMOAnswerBench-Verified at 93.0** show the recipe does not sacrifice general reasoning to get there.

---

## Key takeaways

1. **Pre-training stability is a design choice, not a hope.** The combination of Stable SplitNorm + Muon + µP + a two-stage curriculum that *preserves* a wide LR stability margin is engineering, not accident. See [stable-splitnorm](../pre-training/stable-splitnorm.md).
2. **Context extension to 512K belongs in mid-training.** Pushing the stretch into mid-training rather than a separate long-context SFT stage avoids the usual "lost in the middle" degradation that late-stage extension can introduce.
3. **Agentic coding wants domain-specialised RL, not a monolithic reward.** Per-capability reward routing is what lets a single policy improve on tool-use and reasoning without one washing out the other. See [domain-specialized-rl](../post-training/domain-specialized-rl.md).
4. **MOPD generalises reasoning distillation to a committee.** Instead of one teacher (R1-style), a set of specialist RL teachers on-policy-distill into one student — a template that fits any "multi-skill agent" training problem. See [mopd](../post-training/mopd.md).
5. **Model merging across stages and experts is the final polish.** Not a one-shot soup; a staged, conflict-aware merge over experts and post-training versions.
6. **Sliding-window + full attention at 3:1 is a credible frontier choice.** Keeps most attention cheap while retaining global reach. GQA (48 Q / 8 KV) rather than MLA is the main point of divergence from DeepSeek-V3.
7. **Open-weights, open benchmarks — but the full recipe is where the signal is.** The weights are useful; the three-stage recipe is reusable.

---

## What's still opaque

- **Training-infrastructure specifics** — hardware, pipeline/parallelism topology, and systems-level details (compare DualPipe in DeepSeek-V3) are not detailed in the abstract we worked from.
- **Exact RL algorithm.** "Domain-specialised RL" names the ingredient; the base algorithm (GRPO? PPO? mirror-descent variant?) and the reward mix are not in the abstract.
- **MOPD specifics.** The committee size, teacher-selection policy per prompt, and on-policy distillation loss form are not in the abstract.
- **Model-merging conflict mitigation.** The abstract says conflicts between parameter updates are mitigated; the mitigation mechanism is not spelled out.
- **Training compute.** No GPU-hour figure in the abstract.

---

*Pairs well with:* the [DeepSeek-V3 case study](deepseek-v3.md) for a different point in the "fine-grained MoE, ~15B active, open release" design space — V3 for frontier-cost efficiency and MLA + MTP; IQuest-Q1 for agentic-coding specialisation, 512K context, and the MOPD-driven post-training stack.
