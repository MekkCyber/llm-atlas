# Test-Time Recursive Self-Improvement (TT-RSI)
*Depth — label-free test-time scaling that re-solves each question using the previous full solution as context, with optional in-weight distillation of the chain.*

**TL;DR:** Classical test-time scaling (self-consistency, best-of-N) draws **independent** samples and aggregates them. **TT-RSI** instead draws **recursive** samples: each new attempt is conditioned on the previous *full* solution and asked to refine it. The result is distilled back into the weights via an in-weight step, which can flip an initially wrong majority. On Qwen3 1.7B / 4B / 8B across five competition-math benchmarks, TT-RSI gains **+13.89%** average accuracy over the base model and beats the strongest self-consistency baseline by **+4.79%** at matched test-time compute.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [../rejection-sampling](../rejection-sampling.md)
**Related:** [length-penalty](length-penalty.md) · [orm](orm.md) · [prm](prm.md) · [../rlvr](../rlvr.md)

---

## What it is

A test-time inference recipe with two components, applied at the same compute budget:

- **In-Context RSI (IC-RSI).** For a question $q$, draw $N$ samples sequentially. At step $n$, the prompt is $\{q, o_{n-1}\}$ — the question plus the *previous full solution*. The model is instructed to refine, correct, or build on $o_{n-1}$ rather than re-solve from scratch.
- **In-Weight RSI (IW-RSI).** After a chain $(o_1, o_2, \dots, o_N)$ is produced, treat $o_N$ as the pseudo-label and distill into the weights via SFT on $(q, o_N)$. This can correct an initially-wrong majority — the final refinement in the chain often outvotes the earlier wrong consensus.

Both pieces are **label-free**: no gold answers are used at test time; the chain itself is the training signal.

## How it works

The TT-RSI inference loop for a single question:

```
o_prev = model(q)                       # cold start
for n in 1..N-1:
    prompt = concat(q, o_prev, refine_instruction)
    o_n = model(prompt)                 # recursive refinement
    o_prev = o_n
answer = o_{N-1}
```

The `refine_instruction` directs the model to spot errors, verify intermediate steps, and emit an improved full solution — not just an extension. Compared to self-consistency's $N$ independent samples at temperature $T$, each TT-RSI sample is dependent on the prior.

The in-weight step:

```
dataset = [(q_i, final_o_i) for q_i in prompt_pool]
model.sft_update(dataset)
```

This bakes the test-time chain into the policy, so subsequent single-shot inference on new questions implicitly uses the refined reasoning pattern.

## Why it matters

- **Matches self-consistency's budget, beats its accuracy.** At equal $N$, TT-RSI's recursive sampling uses the same number of forward passes as self-consistency but spends them on refinement rather than voting. Reported **+4.79%** over the strongest self-consistency baseline at matched budget.
- **In-weight distillation closes the wrong-majority gap.** Self-consistency fails when the majority is wrong. The final-step refinement in TT-RSI is empirically more likely to be correct than the majority of earlier steps; distilling that final step transfers its advantage to future inference.
- **Pure label-free.** No verifier, no gold data at test time. Complements verifier-based methods ([../rlvr](../rlvr.md)) and plugs into existing reasoning stacks without new infrastructure.

## Gotchas & tricks

- **Refinement prompts matter.** Generic "please improve your previous answer" prompts underperform explicit "identify the specific errors in your previous solution, then re-derive" prompts. The paper tunes the refinement prompt per domain.
- **Chain length is a tradeoff.** Too short and you recover self-consistency; too long and the chain drifts — later steps sometimes "refine" an already-correct answer into a wrong one. Early stopping on self-agreement between consecutive $o_n$ is a natural stopping rule.
- **IW-RSI amplifies systematic errors.** If the entire chain is wrong for the same reason (shared misconception), distilling the final answer bakes that in. Pair IW-RSI with domain-diverse prompts to dilute shared failure modes.
- **No contamination guard built in.** The in-weight step does not deduplicate against test sets; use [../../data/decontamination.md](../../data/decontamination.md) on the prompt pool before IW-RSI.

## Sources

- Paper: *Beyond Self-Consistency: Recursive Self-Improvement at Test Time* — Wu, Su, Zhang, Li, Wang, Tang, Wang, Ouyang, Tang (Shanghai AI Lab / SJTU), 2026, arXiv 2611.30965.
- Related: *Self-Consistency Improves Chain of Thought Reasoning in Language Models* — Wang et al., 2022 — the baseline TT-RSI beats.
- Related: [long-cot-rl](long-cot-rl.md) — IW-RSI is a label-free sibling of long-CoT RL on refinement trajectories.
