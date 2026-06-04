# Visual RL Environments

*Depth — programmatic generator–verifier environments that produce on-demand visual reasoning training data for RLVR.*

**TL;DR:** Visual reasoning RL has been stuck on static, hand-curated image–question–answer datasets — bounded by collection budget and unable to grow with the model. A visual RL environment replaces the dataset with a generator–verifier program: it samples a fresh latent visual state, renders an image, asks a question, and *exactly* verifies the answer. One pipeline produces an unbounded curriculum at controllable difficulty, giving visual RLVR the same scaling property text-only math/code RL already has.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../evaluation/codeforces-benchmark.md](../evaluation/codeforces-benchmark.md)

---

## What it is

The structural shift:

| | Static visual dataset | Visual RL environment |
| --- | --- | --- |
| Source of (image, q, a) | Pre-collected, fixed | Sampled on demand from a program |
| Verification | Approximate / heuristic | Exact (the generator *knows* the answer) |
| Curriculum | Fixed difficulty mix | Difficulty is a parameter of the generator |
| Scaling | Bounded by collection budget | Unbounded; new instances per step |

A single environment is a 4-tuple:

```
generator()   → latent visual state z
render(z)     → image x
ask(z)        → question q
verify(z, a)  → reward r ∈ {0, 1}
```

TRON (University of Georgia, 2026) instantiates this with 520 environments grouped into five ability buckets — spatial reasoning, mathematical, diagram understanding, pattern/logic, counting.

## How it works

Per training step, with a policy $\pi_\theta$ (a VLM):

```
sample env e ~ env_curriculum
z = e.generator(difficulty=current_level)
x = e.render(z)
q = e.ask(z)
for k in 1..G:
    a_k = sample π_θ(· | x, q)        # G rollouts for GRPO
    r_k = e.verify(z, a_k)
update π_θ with GRPO using (a_k, r_k)
```

Key substrate properties the paper analyzes:

- **Generation reliability** — fraction of generator outputs that render and verify cleanly.
- **Instance and level diversity** — distribution of latent z's at each difficulty.
- **Cross-environment near-duplicates** — programmatic envs can still hit hash collisions.
- **Base-model pass rate by difficulty** — gives the curriculum knob a calibrated mapping.

Two training modes:

- **Single full model** — train on all five ability buckets simultaneously.
- **Per-bucket specialists** — same substrate, train an ability-specialist per bucket. No extra data collection — just route different envs to different policies.

## Why it matters

- **Brings the "infinite verifiable env" pattern to vision.** Same shift that math/code RLVR had — the data ceiling disappears.
- **Curriculum becomes natural.** Difficulty is a generator parameter; you can dial it from the loss signal without re-collecting data.
- **Reproducibility win.** Programmatic envs are versionable and re-runnable in a way curated image datasets are not.
- **Empirical lift.** TRON-DAPO (DAPO on TRON envs) improves Qwen3-VL-4B, Qwen2.5-VL-7B, MiMo-VL-7B across ten external multimodal reasoning benchmarks — not a leak: external benchmarks are held out from the envs.

## Gotchas & tricks

- **Render reliability is the new bottleneck.** Renderer bugs become silent reward-hacking opportunities. Audit verifier rejections per env.
- **Watch for near-duplicates within an env.** Programmatic generators can collapse to small effective state distributions if the latent sampler is biased. Hash images at modest resolution and dedupe.
- **Difficulty calibration drifts as the policy learns.** Recalibrate the difficulty curriculum periodically against base-model pass rates on a held-out probe set.
- **Specialists vs full model is a real tradeoff.** Specialists win on their own bucket; full models generalize. Choose by deployment story.

## Sources

- *TRON: Targeted Rule-Verifiable Online Environments for Visual Reasoning RL* — Yang et al., University of Georgia, 2026 — [arXiv:2606.01599](https://arxiv.org/abs/2606.01599) — primary source for the 520-environment suite and the TRON-DAPO recipe.
- *DAPO* — Yu et al., 2025 — the RL algorithm TRON trains with.
- Analogous text-only "infinite verifiable env" pattern documented in [../evaluation/codeforces-benchmark.md](../evaluation/codeforces-benchmark.md) for code.
