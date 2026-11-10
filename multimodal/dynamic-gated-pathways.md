# Dynamic Gated Pathways (DGP) for MLLMs
*Depth — connect every ViT layer to every LLM layer through learned gates conditioned on the LLM's state.*

**TL;DR:** Multimodal LLMs typically pick one ViT layer's features, run them through a projector, and inject them at a single point of the LLM. DGP replaces that single connection with an `N × M` mesh of gated pathways between ViT and LLM layers: each LLM layer dynamically picks which visual granularity it wants, conditioned on its current hidden state. On LLaVA-1.5 bases, +4.2 on SQA-I and +3.3 on MM-Vet; routing analyses show late LLM layers often pull from *early* ViT features and vice versa.

**Prereqs:** None (self-contained for the ViT↔LLM wiring problem).
**Related:** [_moe](../architectures/_moe.md), [multi-head-attention](../architectures/multi-head-attention.md)

---

## What it is

The dominant MLLM pattern (LLaVA, MiniGPT-4, Qwen-VL, InternVL) picks a *single* ViT layer, projects its tokens through a small MLP ("the projector"), and inserts them into the LLM as a prefix of extra tokens. The choice of which ViT layer to tap is a tuning headache:

- **Early layers** carry low-level visual detail (texture, edges, color).
- **Middle layers** carry object parts and local semantics.
- **Late layers** carry global scene semantics and are the most abstracted.

Different downstream tasks prefer different granularities, but the single-projector design forces you to pick one in advance. Multi-layer injection schemes exist but typically fix the routing in advance.

DGP turns the choice into a learned, per-token, dynamic routing: every LLM layer receives a weighted mix of features from every ViT layer, with weights computed from the LLM layer's own hidden state at that position.

## How it works

### The mesh

Let the ViT produce features at `N` layers: `V_1, …, V_N`. Let the LLM have `M` transformer blocks. Build an `N × M` bank of projectors `P_{i,j}: d_vit → d_llm`. (In practice, parameters are shared across subsets of pairs to keep the mesh small.)

### Per-layer gating

At each LLM layer `j`, before the block operates, compute a gating vector `g_j ∈ Δ^N` from the LLM's hidden state at that layer:

```
g_j = softmax( MLP_j(h_j) )              # mixture over ViT layers
v_inject_j = Σ_i g_{j,i} · P_{i,j}(V_i)   # mixed visual features at LLM-layer j
```

The mixed visual features are added to (or concatenated with) the LLM layer's visual tokens. Because `g_j` depends on `h_j`, routing is dynamic per position and per query.

### Training

Standard VLM pretraining + instruction-tuning stages, with the DGP gates and projectors trained jointly with the LLM. The mesh adds a small parameter overhead relative to the full MLLM.

### What the routes actually learn

The paper reports that gates settle into **non-trivial inverted patterns**: late LLM layers (which produce the final output) often attend to *early* ViT layers (low-level detail), while early LLM layers (which still build intermediate representations) attend to *late* ViT features (global semantics). The intuition: late LLM output work benefits from grounding in concrete visual detail, not more abstraction.

## Why it matters

- **Removes a brittle architecture choice.** "Which ViT layer should I project?" was one of the top three MLLM tuning levers. DGP replaces it with a learned, per-token mesh.
- **Same base, better numbers.** On LLaVA-1.5-7B baselines, +4.2 SQA-I, +3.3 MM-Vet, with gains over prior multi-layer fusion methods. Similar gains at 13B.
- **Transferable routing insight.** The "late LLM wants early ViT" finding is useful even if you don't adopt DGP — it argues against the common practice of injecting *only* late ViT features.
- **Structurally akin to MoE routing.** The gating-over-inputs pattern is a visual analogue of expert routing in MoE. Techniques from MoE (load balancing, capacity constraints, aux losses) may translate directly.

## Gotchas & tricks

- **Compute and memory scale with the mesh.** `N × M` projectors is too many at full parameter count — share parameters across groups (e.g., tie all `(i, j)` for `i` in a quartile of ViT depth).
- **Gate collapse.** Without any regularization, gates can collapse to a single ViT layer per LLM layer, defeating the whole point. Add an entropy bonus on `g_j` or an MoE-style load-balancing loss.
- **Projector sharing matters.** Projectors that are too shared lose per-pair specialization; too independent and the model over-fits with few training tokens. The paper's recipe shares across LLM-layer groups.
- **Doesn't fix a weak ViT.** If the vision backbone's features are poor, no amount of clever routing recovers them. Pair DGP with a modern ViT (CLIP/SigLIP/InternViT-class).
- **Inference cost.** At serve time, the mesh means every ViT layer's features must be kept available (not just the final layer). This increases vision-feature memory by `N×`. For long prompts this is non-trivial.

## Sources

- Paper: *Where to Connect? Boosting MLLMs via Dynamic Gated Pathways across ALL ViT and LLM Layers* — Yan, Tang, Wei, Wang, Zhang, Chen, Xia, Geng, Wu, Chen, 2026 — arXiv 2611.06844.
- Paper: *LLaVA: Visual Instruction Tuning* — Liu, Li, Wu, Lee, 2023 — the single-projector MLLM baseline DGP generalizes.
- Paper: *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks* — Chen et al., 2023 — multi-stage visual-language training.
