# RLVR — RL from Verifiable Rewards
*Depth — use programmatic verifiers, not a learned reward model, as the reward signal.*

**TL;DR:** Run PPO (or GRPO) post-training where the reward for each rollout is a binary **ground-truth check**: does this math answer match the reference? Do these code tests pass? Does the output follow the required format? No reward model, no human preference labels during the RL phase. Cheaper to run, robust to reward hacking, but only applicable to domains where correctness is machine-verifiable. Core post-training ingredient for Tülu 3, DeepSeek-R1, and every modern reasoning-focused model.

**Prereqs:** [transformer-block](../architectures/transformer-block.md), [_rl](_rl.md)
**Related:** [_post-training](_post-training.md), [_rewards](_rewards.md), [grpo](grpo.md), [long-cot-rl](reasoning/long-cot-rl.md), [orm](reasoning/orm.md), [prm](reasoning/prm.md), [olmo-2 case study](../case-studies/olmo-2.md), [deepseek-r1 case study](../case-studies/deepseek-r1.md), [composer2](../case-studies/composer2.md)

---

## What it is

Classical RLHF has three stages:

```
1. SFT — supervised fine-tune on curated (prompt, response) pairs.
2. Train a reward model (RM) on human preference data (A vs. B labels).
3. PPO — optimize the policy to maximize RM's score, with KL penalty to SFT model.
```

The reward model is the weak link: it's trained on finite human preferences, can be *hacked* by the policy (find outputs the RM scores highly but a human wouldn't), and is expensive to serve in the training loop.

**RLVR replaces step 2 entirely.** The reward function is a **programmatic verifier**:

```
reward(prompt, response) ∈ {0, 1}   # or scalar in [0, 1]
```

where `reward` is deterministic code that checks some verifiable property:

- Math: does `extract_final_answer(response) == reference_answer`?
- Code: do the unit tests pass?
- Format: does the response match the required schema (XML, JSON, "ANSWER: ...")?
- Instruction following: does the response satisfy the constraints stated in the prompt (length limit, language, contains keyword)?

The policy update is standard PPO or GRPO; only the reward source changes.

## How it works

### The training loop

```
for step in range(n_steps):
    prompts = sample_from_dataset()
    responses = policy.generate(prompts)              # K rollouts per prompt
    rewards = [verifier(p, r) for p, r in zip(prompts, responses)]
    advantages = compute_advantages(rewards)          # GRPO: group-relative
    policy.update(prompts, responses, advantages,
                  kl_penalty_to_reference=β)
```

No reward model. No human-in-the-loop per step. The verifier is a pure function of (prompt, response), usually running in milliseconds on CPU.

### Why a binary signal is enough

Sparse reward seems weak, but for reasoning tasks it works remarkably well because:

1. **Many rollouts per prompt.** GRPO or best-of-K sampling generates 8–64 responses per prompt. Even if the success rate is 10%, you get useful positive signal on some rollouts.
2. **Group-relative advantages.** GRPO normalizes rewards within the group of K rollouts for the same prompt. Even if all rollouts are binary, the *relative* ranking is a rich signal.
3. **The verifier is perfectly aligned with the desired capability.** No reward model bias, no OOD surprises. If the verifier says the answer is right, it is right.

### Combining with SFT and DPO

RLVR is usually the last step, not a replacement for SFT or DPO:

```
Base → SFT (teach format + broad behavior)
     → DPO (preference-shape style + refusal)
     → RLVR (sharpen verifiable capabilities: math, code)
```

The earlier stages put the model in a reasonable output distribution. RLVR pushes it to actually *solve* the verifiable problems within that distribution. Skipping SFT and doing RLVR from base is possible — **DeepSeek-R1-Zero** does exactly this on `DeepSeek-V3-Base`, with only rule-based accuracy + format rewards, no demonstrations, and still drives AIME 2024 pass@1 from **15.6% → 77.9%** (see [long-cot-rl](reasoning/long-cot-rl.md)). It needs more rollouts and a strong base, and the emergent CoT is hard to read, but it *works* — which is the main surprise from R1.

### Verifier design is the ML work

The algorithm is boring PPO/GRPO — the **ML judgment goes into the verifier**. Good verifier design:

- **Strict enough to prevent hacking.** If the verifier accepts "answer: 42" and "Answer: 42.0" and "42", the model may learn odd capitalization tricks. Normalize answers before comparison.
- **Lenient enough to not reward only one format.** If the verifier only accepts exact string match, the model won't learn to explain its reasoning — it learns to emit bare numbers.
- **Robust to prompt-injection-style outputs.** The model may try to write "The answer is X" for *every* X to game pattern matches. Extract the final answer from a specific tag or position.
- **Cheap.** The verifier runs on every rollout, often 10s of thousands per step. A 1-second check becomes a bottleneck.

## Why it matters

- **Removes the most hackable component of RLHF.** The learned reward model is the single biggest source of alignment tax in classical RLHF. RLVR ditches it.
- **Cheaper per step.** No RM forward pass, no RM training cost, no RM serving infra.
- **Natural fit for reasoning capability gains.** Math and code have ground truth. Running RLVR on math prompts is *the* recipe for improving math scores on a model that's already been SFT'd.
- **Composes with self-play / bootstrapping.** Since the verifier is programmatic, you can generate new prompts (teacher model writes a math problem, verifier checks the answer), build infinite curriculum without human labels.

## Gotchas & tricks

- **Only works for verifiable domains.** You cannot do RLVR for "is this response helpful" or "is this tone appropriate" — those require human judgment. Pair RLVR with DPO for the non-verifiable style/refusal axes.
- **Reward sparsity at init.** If the pre-RLVR model can't solve *any* of the problems, every reward is 0 and gradients are zero. Use SFT beforehand to bootstrap success rate above ~5–10%, or start with easier problems and curriculum up.
- **KL penalty still needed.** Without it, the policy can drift off-distribution to maximize the narrow verifier signal, losing general capabilities. β in the 0.01–0.1 range is typical.
- **Watch for verifier gaming.** Monitor a held-out eval where the verifier is stricter (or human-judged) than the training verifier. Divergence between training-verifier reward and held-out quality is the canary.
- **Mix verifier types in one run.** Math verifier + code verifier + format verifier, chosen per prompt by dataset tag. Keeps the model from overspecializing on one signal.
- **GRPO > PPO here.** GRPO's group-relative advantage removes the need for a value model — with binary rewards and K rollouts per prompt, there's nothing for a value model to learn that the group mean can't give you.
- **Don't confuse with "process reward models".** [PRM](reasoning/prm.md) is a different technique: learn a reward model on *steps of reasoning*, not just final answers. Orthogonal to RLVR; can be combined. DeepSeek-R1 tried PRMs and abandoned them for RL — reward hacking dominated.
- **Outcome-only rewards are surprisingly strong.** Uesato 2022 showed that on GSM8K, RL with outcome-only reward (equivalent to RLVR) ties RL with step-level PRM rewards. The common intuition that "dense rewards must help" is wrong in this regime — a good base model has enough implicit reasoning structure that outcome rewards suffice. This is the empirical foundation for every RLVR-based reasoner from Tülu 3 onward.
- **Composite rewards are the norm.** RLVR in production is `accuracy + format + (sometimes) language-consistency`, direct-summed. See [_rewards](_rewards.md) for the full picture of how reward components combine.

## Sources

- Paper: *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* — AI2, 2024 — names and formalizes RLVR, provides the canonical open recipe.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — uses verifiable rewards (with GRPO) at scale for reasoning; R1-Zero does it without any SFT, R1 uses them in Stages 2 and 4 of a 4-stage pipeline.
- Paper: *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — DeepSeek, 2024 — introduces GRPO, which is the standard policy optimizer used with RLVR.
- Paper: *Training Verifiers to Solve Math Word Problems* — Cobbe et al., 2021 — early GSM8K verifier, precursor to the "use correctness as the signal" framing (see [orm](reasoning/orm.md)).
- Paper: *Solving Math Word Problems with Process- and Outcome-Based Feedback* — Uesato et al., 2022 — shows outcome RL ≈ process RL on GSM8K, foundational for RLVR's confidence.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — rejection sampling + verifiable checks for math/code in post-training (same spirit).
