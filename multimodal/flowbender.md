# FlowBender
*Depth — feedback-aware training that promotes inference-time constraint violation to a first-class network input.*

**TL;DR:** Conditional diffusion and flow models routinely fail the constraints that define their task. FlowBender treats the constraint-violation signal as an input the network sees during training, so the model learns a **correction policy** conditioned on inference-time feedback — turning external inference-time guidance into an integrated correction step. Gilo et al., Technion + NVIDIA + U Toronto + Vector, arXiv 2606.20404. Beats supervised, alignment-loss-augmented, and SOTA inference-time guidance baselines.

**Prereqs:** [README](README.md)
**Related:** none yet (no flow-matching coverage in the graph)

---

## What it is

A closed-loop training scheme for conditional generative flows. At each step, evaluate the constraint (e.g. "does the generated image contain the required object", "does the generated layout match this spec") on the current sample, feed the violation back into the network as an additional input, and train the network to emit the correction.

The conventional split — train without enforcing constraints, then add guidance at inference — leaves a train/test mismatch: the model has never seen "what to do when you're violating the constraint." FlowBender closes that loop.

## How it works

For a conditional flow $v_\theta(x_t, c, t)$ that maps a noisy sample $x_t$ at time $t$ under condition $c$ to a velocity:

1. **Forward sample.** Run a few steps of the flow to produce a candidate $\hat{x}$.
2. **Evaluate the constraint.** Compute a violation signal $g(\hat{x}, c) \in \mathbb{R}^d$ — a vector describing how the sample fails to meet the condition.
3. **Augment the network input.** Re-run the network as $v_\theta(x_t, c, t, g)$ — the violation is just another conditioning input.
4. **Supervise on the correction.** Train so that the augmented velocity moves $x_t$ toward a sample with lower violation.

At inference, the same loop runs: generate, evaluate violation, feed back, refine. No external guidance solver, no per-sample optimization — the correction is learned in the weights.

The contrast with classifier-free guidance and alignment losses:
- **CFG** modifies the velocity at inference using an external classifier or unconditional model; cheap to add, expensive to run, brittle to constraint type.
- **Alignment loss** during training penalizes constraint violation but doesn't condition the network on it — the network has no input encoding "you are currently violating constraint X."
- **FlowBender** does both: penalizes violation *and* gives the network a violation input to condition on.

## Why it matters

- **Closes the train/test gap** for conditional generation: the network now sees constraint violation as a normal input class, not an external signal.
- **No inference-time guidance cost.** Inference is a plain forward pass; correction is in the weights.
- **Generalizes** to any setting where the constraint is cheap to evaluate at training time — structured outputs, text-to-SQL, code, layout generation.

## Gotchas & tricks

- **Violation signal design is the load-bearing choice.** A scalar "constraint met / not met" is too coarse; the violation needs to be a vector that varies smoothly with how the sample fails.
- **Differentiability not required.** The violation can be from a non-differentiable checker (rule-based, simulator), since the network is trained to predict corrections, not to backprop through the checker.
- **Bootstrap problem.** Early in training the network's samples violate everything; useful violation signals only emerge once samples are partially correct. Curriculum (start with easy constraints) helps.
- **Compatible with CFG.** FlowBender doesn't replace classifier-free guidance — it's orthogonal and can be stacked on conditional inputs.

## Sources

- Paper: *FlowBender: Feedback-Aware Training for Self-Correcting Conditional Flows* — Gilo, Elflein, Sobol, Litany, Technion + NVIDIA + University of Toronto + Vector Institute, 2026, arXiv 2606.20404.
