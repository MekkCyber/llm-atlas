# SecOPD — Secure On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Existing defensive fine-tuning against prompt injection (StruQ, SecAlign, Meta-SecAlign) uses **sequence-level** feedback (DPO or GRPO), which treats an entire response as equally (in)secure and can't say *which tokens* were the problem. **SecOPD** trains with **token-level feedback** by having the model produce a rollout on the injected input and scoring each token by the *initialization model's* probability under the *clean* (uninjected) input. Introduced by Peng, Lian, Wagner, Chen 2026; drops adaptive-attack ASR from 94.0% (Meta-SecAlign SoTA) to 9.0% on Qwen3.6-27B.

**Prereqs:** [_attacks.md](_attacks.md), [../post-training/dpo.md](../post-training/dpo.md)
**Related:** [_prompt-injection-defenses.md](_prompt-injection-defenses.md), [payload-splitting.md](payload-splitting.md), [prefix-injection.md](prefix-injection.md)

---

## What it is

Prompt injection is the #1 threat to deployed agents: attackers embed prompts inside data ("ignore all prior instructions and X…") that the agent later reads and executes. Defensive fine-tuning tries to teach the model to refuse this pattern, but adaptive attacks — attacks that know the defense recipe — kept driving ASR back to near 100% against the strongest defenses.

SecOPD's diagnosis: prior defenses treat responses as atomic (correct or wrong), which is fine for standard SFT/RLHF but insufficient for prompt injection, where a response is often *mostly* correct with a small vulnerable substring. The learning signal is too coarse.

## How it works

### The distillation loop

For each defense example, you need two prompts:

- **Injected input** — the attack-carrying prompt the model must resist.
- **Clean input** — the same prompt without the injection payload.

Then:

1. The model (being defended) produces a **rollout** on the injected input.
2. For each token in that rollout, compute the **initialization model's** probability of that token under the *clean* input.
3. That per-token clean-input probability is the distillation target. Fine-tune the defended model to match the clean-input distribution at every token of its own injected-input rollout.

### Why this is token-level defense

The defended model is essentially learning: "at each token position, produce what a safe model would produce on the clean input, even though I'm reading the injected input." Tokens the injection tried to hijack now get supervised toward the clean behavior at that specific position, not against a blob-level judgment.

### Adaptive-attack robustness

Adaptive attackers optimize the injection to exploit specific weaknesses in the defense. Sequence-level defenses have one bit of feedback per response — very few gradients to hide behind. Token-level defenses give the model dense, per-token gradients aligned with clean behavior, drastically shrinking the surface an adaptive attacker can exploit.

## Why it matters

- **>10× ASR reduction on the SoTA adaptive benchmark.** 94.0% → 9.0% against PISmith on Qwen3.6-27B. On agentic tool-calling (unseen domain), 4.7% vs 5.5% for Meta-SecAlign — generalization holds.
- **Reusable per-token distillation lens.** The clean-vs-injected token comparison is a specific instance of a broader idea: turn any "output is bad" signal into per-token gradients by comparing to a reference model. Same lens applies to jailbreak defense, toxicity mitigation, and safety alignment more broadly.
- **Doesn't require preference data.** Where DPO needs paired (chosen, rejected) responses, SecOPD needs (injected input, clean input) prompt pairs — much cheaper to synthesize.

## Gotchas & tricks

- **Clean-input pairs must actually be clean.** If the "clean" input still contains a subtle instruction ambiguity, the reference distribution learns wrong behavior. Verify clean inputs with a small held-out check.
- **Initialization model choice matters.** The reference is the *initialization* model — the pre-defense snapshot. If that model already has vulnerabilities, SecOPD amplifies them. Use a well-behaved base or run one round of standard defensive SFT first.
- **Token-level distillation preserves teacher patterns.** If the initialization model has odd verbosity or formatting on clean inputs, SecOPD copies that pattern faithfully. Distill from a version whose *style* you're comfortable inheriting.
- **Doesn't defend against attacks that don't produce a visible per-token divergence.** If the attack manipulates upstream state (e.g. memory poisoning across turns) rather than producing a distinguishable token, per-token clean-input comparison won't catch it.
- **KL to reference is implicit.** Distilling to the reference *is* a KL constraint. If you also add an explicit KL penalty, you're double-regularizing — tune down or drop.

## Sources

- Paper: *SecOPD: Mitigating Adaptive Prompt Injections by On-Policy Distillation* — Peng, Lian, Wagner, Chen, 2026. [arXiv:2608.21500](https://arxiv.org/abs/2608.21500). Code: [github.com/pppyb/SecOPD](https://github.com/pppyb/SecOPD).
- Related: *Meta-SecAlign* — the prior SoTA defense SecOPD compares against.
- Related: *StruQ* — structured queries for prompt-injection defense; a sequence-level defense.
