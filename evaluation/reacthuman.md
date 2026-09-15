# ReactHuman
*Depth — a physics-grounded benchmark for reactive decision-making by embodied multimodal LLMs, with deterministic-simulation ground truth.*

**TL;DR:** Existing embodied-LLM evaluations either probe intuitive physics passively (Q&A over videos) or measure deliberate long-horizon planning (navigation, rearrangement) — neither tests whether a model can turn physical understanding into immediate safety-critical action. ReactHuman is the first physics-grounded reactive-decision benchmark: the MLLM acts as the brain of a simulated humanoid facing sudden household hazards (catching a slipping plate, dodging a falling knife). 17 event families, >1,000 bit-for-bit reproducible scenes from 240 Hz rigid-body simulation, five-metric scoring along three axes (reasonable, safe, physically grounded). Every committed plan is physically executed. 7 evaluated MLLMs mishandle roughly one hazard in three; none of the failures shrink with model scale.

**Prereqs:** *(none in-graph yet)*
**Related:** [mmlu](mmlu.md)

---

## What it is

Deploying MLLMs as the decision core of household robots requires *reactive safety*: the model must, within milliseconds to a second, choose an action that avoids or intercepts a suddenly-emerging physical hazard. Traditional evaluations probe adjacent capabilities but miss this exact test — a passive video Q&A benchmark cannot tell you whether the model would produce the right *action* when a plate slips.

ReactHuman's design principle: **committed plans must have observable physical consequences.** The evaluation harness physically executes what the model commits to, then scores based on the resulting simulation state.

## How it works

### Scenario generation

- **17 event families** — falling objects, sliding items, collision trajectories, tipping hazards, and other sudden household perturbations.
- **>1,000 scenes**, each **bit-for-bit reproducible** through deterministic simulation at 240 Hz rigid-body physics. No annotation is required — ground truth is derived from the simulator's own state trajectory.
- **Adversarial objects.** Some scenes include objects whose *appearance contradicts their physics*: a foam anvil (looks heavy, is light), a steel apple (looks light, is heavy). These test whether the model reasons from *motion cues* or from *appearance priors*.

### The evaluated system

The MLLM sits at the decision core of a simulated humanoid. It receives the scene observation (rendered from the simulator), the current state, and must emit a committed plan (target action + interception point + timing). The plan is physically executed inside the simulator; the resulting scene state is what gets scored.

### Five-metric suite along three axes

Actions are scored along three axes:

1. **Reasonable** — is the chosen action a sensible response to the observed hazard?
2. **Safe** — does executing the plan avoid harm to the humanoid, the environment, or the object?
3. **Physically grounded** — did the plan respect physics (right interception point, right timing, right force magnitude)?

Five distinct metrics decompose these axes into checkable predicates; the aggregate suite gives per-axis and overall scores.

## Why it matters

- **Fills a real gap.** Reactive safety was untested; every prior embodied-LLM eval measured a different capability.
- **Deterministic ground truth.** Simulator state → labels means the benchmark is *training-signal-eligible* — a scalable reward source for RL / RLVR-style post-training of embodied models, not just an eval.
- **Model scale doesn't fix these failures.** The paper reports 7 representative MLLMs mishandle ~1 in 3 hazards, act from fixed dispositions rather than the observed scene, and miss interception points at meter scale — *none of these shrink with scale*. That's a strong "reactive safety needs a training-signal, not a bigger base" signal.
- **Adversarial objects operationalize the intuitive-physics gap.** Foam anvil / steel apple explicitly probe whether models trust appearance over motion; a failure here can't be dismissed as a perception artifact.

## Gotchas & tricks

- **Ground truth requires the simulator's rules.** ReactHuman labels are *simulator-consistent*, not real-world-consistent. If a model's action depends on real-world physics beyond the sim (aerodynamics, friction gradients), the simulator's answer may not transfer.
- **Timing granularity is 240 Hz.** Actions committed at model-inference-latency (100s of ms) must anticipate rather than react — the paper's five-metric suite scores this via *interception point at meter scale*, which is intentionally strict.
- **Adversarial objects are the sharp diagnostic.** Reporting per-family scores conceals the "appearance-over-motion" failure; report the adversarial-object cell separately.
- **Not a substitute for real-world evaluation.** Sim-only success is necessary but not sufficient; a model that passes ReactHuman may still fail on real robots due to perception artifacts.

## Sources

- Paper: *ReactHuman: A Physics-Grounded Benchmark for Human-Like Reactive Decision-Making in Embodied Multimodal LLMs* — Li, You, Xiong, Chen, Zhao, Wu, Zhang, Liu, 2026 — [arXiv:2609.10895](https://arxiv.org/abs/2609.10895).
