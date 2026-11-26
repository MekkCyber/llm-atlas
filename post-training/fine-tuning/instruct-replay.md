# InstructReplay
*Depth — self-generated-replay regularization against catastrophic forgetting during factual fine-tuning.*

**TL;DR:** Fine-tuning an instruction-tuned LLM on new facts collapses its general capabilities (−35 points on average across benchmarks, plus answer-length collapse). InstructReplay freezes a snapshot of the initial model's answers to a general-instruction pool and uses them as replay targets during fine-tuning. Reduces the average decline on five general benchmarks from 28.1 → 2.4 points while learning new facts about as well as standard SFT. Across three model families and two factual domains.

**Prereqs:** [post-training/README.md](../README.md)
**Related:** [model-souping.md](../../pre-training/model-souping.md)

---

## What it is

A simple, model-specific regularizer against catastrophic forgetting in fine-tuning. Instead of adding KL to a reference model (which helps but costs a forward pass per step), InstructReplay generates a reference *dataset* once from the initial model's answers to a pool of general instructions, then fine-tunes with a mixed loss: new-fact SFT on the task data plus plain SFT against the pre-recorded self-answers on the replay data. The replay is the model's own prior output — so the target is always reproducible and model-specific, no external reference needed at train time.

## How it works

1. **Build the replay pool.** Pick a general-instruction set (e.g., Alpaca / UltraChat / in-house helpful-instruction mix). ~10K–100K prompts.
2. **Snapshot answers.** Before fine-tuning, run the pre-training-complete model over the replay pool once and save its answers verbatim.
3. **Mixed-batch fine-tuning.** Each batch: some fraction `α` of new-fact data (SFT target = the new fact's answer), the rest from the replay pool (SFT target = the saved self-answer). `α ≈ 0.5` is a working default.
4. **Standard SFT loss** on both halves. No KL, no special tricks beyond the data mix.

The replay targets are just teacher-forced strings — tokenize once, train normally.

## Why it matters

- **Trades disk for stability.** Pre-recording is O(replay_pool_size) cheap; at train time, replay data looks like any other SFT data. No extra forward pass per step.
- **Specific to this model's behavior.** Replaying *this* model's prior answers (not a generic reference) preserves this model's exact tone, length, and formatting.
- **Measured effect is large.** Average decline 28.1 → 2.4 points, across three families and two domains. The length collapse (models answering "yes" when they used to answer in paragraphs) goes away too.
- **Drop-in for existing SFT pipelines.** No new loss, no new hyperparameters beyond the mix ratio `α` and the replay pool size.

## Gotchas & tricks

- **Replay-pool coverage matters.** If the replay pool is small or distributionally narrow, you anchor only on those behaviors. Diverse pools (mixed task types, lengths, languages) preserve more.
- **`α` is the main knob.** Too low (`α ≈ 0.9`): new-fact learning drifts the model and the small replay share can't anchor. Too high (`α ≈ 0.1`): new facts are under-weighted. The paper sweeps; ~0.5 works.
- **Snapshot drift.** The replay targets are frozen *before* fine-tuning. If you iterate on fine-tuning recipes, re-snapshot — otherwise replay anchors on a stale model.
- **Not helpful for RL post-training.** The method targets factual-SFT collapse; it doesn't address RL-driven collapse (where KL to a reference model is still the right tool).
- **Composes with LoRA.** Nothing requires full-parameter fine-tuning; apply with LoRA on new facts and the regularization still works, because replay is just more SFT data.
- **Watch for contamination of the replay pool.** If the replay pool overlaps the new-fact domain, replay reinforces the old (possibly wrong) answer and competes with the fine-tune.

## Sources

- Paper: *InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning* — Shi, Kou, Zhou (BAAI), 2026 — [arXiv:2611.32918](https://arxiv.org/abs/2611.32918).
