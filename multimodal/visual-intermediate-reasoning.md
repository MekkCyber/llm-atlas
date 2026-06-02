# Visual Intermediate Reasoning

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reasoning substrate for Vision-Language-Action (VLA) policies that replaces text chain-of-thought with *visual evidence tokens* — compact spatial intermediates emitted before the action head. Text CoT for embodied control adds multi-second decode latency and discards spatial precision; visual intermediates preserve spatial cues while avoiding autoregressive decode overhead. With selective routing of which visual-evidence tokens to emit, step latency on BridgeData V2 drops from 8.4s (text CoT) to 0.37s — a 22.8× speedup.

**Prereqs:** *(assumes familiarity with VLA policies and chain-of-thought reasoning)*
**Related:** [README](README.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Recent VLA work equips robot policies with explicit reasoning steps before action prediction, on the intuition that single-shot action decoding leaves capability on the table. The default substrate has been text chain-of-thought (ECoT and successors): the model emits a textual reasoning trace, then conditions the action head on it.

Two problems for embodied control:

1. **Latency.** Text CoT is autoregressive — tens to hundreds of tokens × ~50ms/token = multi-second decode. Closed-loop robotic control at this latency is non-viable.
2. **Representation mismatch.** Spatial relationships (object pose, contact points, grasp orientation) lose precision when projected through natural-language tokens. Worse, irrelevant or hallucinated text actively confuses the action head.

Visual Intermediate Reasoning swaps the substrate. The model emits *visual* evidence tokens — a small set of grounded spatial features — and the action head consumes those directly. Spatial precision is preserved; decode latency collapses to a single short forward pass.

---

## How it works

### Visual evidence tokens

The reasoning step produces a small set of visual tokens (the paper calls them visual evidence tokens) that carry grounded spatial information about the scene relevant to the next action. The exact representation is a learned set of feature tokens projected from the vision encoder's spatial features — interpretable in the sense that they correspond to visual regions/objects rather than to language.

### Selective routing

Not every action step needs the full set of visual tokens. A *selective routing* mechanism learns which tokens to emit per step — a sparse, task-conditioned subset. This is the latency lever: emitting few tokens cuts the reasoning-step cost; emitting many gives more spatial conditioning. The router is trained end-to-end with the policy.

### Supervision via VisualEvidence-Kit

The paper releases VisualEvidence-Kit, a 754.7K-instruction VLA supervision dataset built by a "VisualEvidence-Agent" that generates intermediate-evidence labels for action data, plus counterfactual faithfulness tests (perturbing the visual evidence and checking the action changes appropriately). Routing supervision and faithfulness audits both come from this set.

### Decoupled from text CoT

Visual intermediate reasoning is *not* layered on top of text CoT — it replaces it. The model can still consume text instructions at the input, but reasoning between perception and action runs in visual-token space, not text-token space.

---

## Why it matters

- **Makes reasoning-augmented VLAs real-time.** 22.8× speedup on the headline benchmark moves reasoning-augmented closed-loop control from the simulation regime into the wall-clock regime.
- **Preserves spatial precision.** The information that gets lost in text-CoT projection (poses, distances, contact geometry) survives in visual tokens.
- **Generalizes the "intermediate-tokens-as-thinking" pattern.** If text CoT helps text reasoning by letting the model write down structured intermediates, *visual* CoT helps spatial reasoning by the same logic. Likely to spread across embodied AI and physical-action models.

---

## Gotchas & tricks

- **Don't measure latency end-to-end without isolating the reasoning step.** The 22.8× number is the *reasoning step* alone; total inference also includes encoder and action-head passes that don't change.
- **Routing supervision is critical.** Without it the router collapses (always all-on or all-off) and either latency or quality goes. The counterfactual faithfulness tests in VisualEvidence-Kit are the practical fix.
- **Vision-encoder dependence.** Visual evidence tokens inherit whatever the vision encoder didn't extract. A weak encoder caps the reasoning quality regardless of how the action head is trained.
- **Not a substitute for high-level planning.** Visual intermediate reasoning handles the *perception-to-action* gap. Long-horizon task decomposition still needs a separate planner.

---

## Sources

- Paper: *VisualThink-VLA: Visual Intermediate Reasoning for Effective and Low-Latency Vision-Language-Action Policies* — Zhang, Yuan, Dai, Yu, Lv, Zheng, Zhu, Ge, Wan, Tang, Zhuang, 2026 — Zhejiang U. / Cornell / NUS / XIDIAN. Introduces visual evidence tokens, selective routing, and VisualEvidence-Kit; reports 22.8× reasoning-step speedup on BridgeData V2.
