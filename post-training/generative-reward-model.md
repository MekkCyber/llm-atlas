# Generative Reward Model (Vocabulary-Space Readout)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of adding a scalar classifier head to score candidate outputs, use the base LM's own **vocabulary space** as the readout channel. Project the model's logits onto a designed "verification" axis (e.g., equivalence probabilities distilled from an offline oracle) and take the resulting continuous score as the reward. Reference-free, generalizes zero-shot to unseen distributions, and — because it's the same forward pass as the base model — decision-projected logit lens + SAEs give *free error localization*. Introduced by **GenV** (2026), which reached 0.961 AUROC on autoformalization verification and +11.3 pts on downstream agentic accuracy.

**Prereqs:** [_rewards.md](./_rewards.md), [reasoning/orm.md](./reasoning/orm.md)
**Related:** [reasoning/prm.md](./reasoning/prm.md), [../interpretability/README.md](../interpretability/README.md)

---

## What it is

Two families of reward models dominate today:

- **Discriminative RMs**: a scalar head on top of a base LM, trained on preference or correctness pairs. Fast, but throws away everything except a single number and typically forgets the base model's world knowledge outside the fine-tuning distribution.
- **LLM-as-judge**: prompt an LLM to output a verdict token or JSON; use its token probability as the score. Preserves world knowledge but is fragile to prompt phrasing and expensive to distill.

A **generative reward model** does neither. It repurposes the base LM's *native* vocabulary space as the score readout — no new head, no verdict prompt. The reward is a continuous projection of the model's logits onto a chosen axis in vocabulary space that has been aligned (via distillation from an oracle) with the property you want to score.

## How it works

Concretely, for GenV in the autoformalization setting:

1. **Oracle-mine equivalence pairs.** Run Z3 as an offline reference-equivalence oracle over many candidate formalizations; produce a supervised set of (candidate, is-equivalent) labels.
2. **Distill into the LM's vocabulary space.** Fine-tune such that a designed linear direction in logit space (a "decision projection") aligns with the equivalence probability. This does not add a head; it re-shapes the softmax.
3. **Score at inference.** Given a new candidate, run one forward pass and project the logits onto the decision axis. That scalar is the reward.
4. **Free localization.** Because scoring lives in the base model's own token space, decision-projected logit lens and sparse autoencoders on intermediate activations *natively* highlight *where* the candidate errs — no separate localization head needed.

Hard-negative mining (GenV+HN) sharpens the axis by training on adversarial near-equivalents.

## Why it matters

- **Zero-shot transfer.** GenV+HN generalizes to unseen translators and formal styles without re-training. Discriminative RMs on the same task collapse under distribution shift.
- **Interpretability for free.** SAEs / logit lens on the readout dimension give per-token attributions of the reward, exposing which parts of the candidate drive the score. This is essentially impossible with a black-box scalar head.
- **New failure mode named: VPU.** GenV was motivated by the observation that verdict-only checks (solver SAT/UNSAT match) can be provably at chance level on **Verdict-Preserving-Unfaithful** translations — the reward model has to look at the *translation*, not just the verdict. Applies well beyond autoformalization.

## Gotchas & tricks

- **Axis design is a first-class problem.** The vocabulary-space projection has to align with the property you want; badly-chosen axes collapse to random. GenV uses an offline oracle to specify the axis.
- **Distillation objective, not classification.** The training loss is a distillation into the LM's own soft outputs, not cross-entropy on a discrete label. Preserves the base model's calibration.
- **Reference-free at inference.** No target formalization needs to be provided at scoring time — the axis carries the "reference" information implicitly.
- **Not the same as LLM-as-judge.** LLM-as-judge prompts the base model with an instruction and reads a verdict token. GenV re-shapes the model itself and reads a *continuous* projection, avoiding both prompt sensitivity and the discreteness of the verdict channel.
- **Requires an oracle.** GenV needs an offline reference-equivalence oracle to mine training targets. For domains without a solver, this reduces to standard RM training on preference labels.

## Sources

- Paper: *Beyond Solver Verdicts: Generative Reward Models for Autoformalization* — Singh, Ganguly, Goel, Torkamani, Han, Lilien, Erata, Chaudhary (CWRU, Yale, Argonne), 2026 — arXiv:2609.11085.
