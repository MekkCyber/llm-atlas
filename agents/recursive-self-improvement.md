# Recursive Self-Improvement (Recursa)
*Depth — adapt a frozen foundation model to new domains by growing an auditable knowledge repository through parallel-learner self-improvement cycles.*

**TL;DR:** A **system-level** adaptation method: the base model weights never change, but a structured external **knowledge repository** (domain skills + application conditions + reflections on past tasks) grows by validated self-improvement cycles. **Parallel learners** each try a task, propose local improvements, and a validator merges accepted entries into the shared repository. "Recursive" because next-iteration workers read the updated repository. On Harvey LAB's Firm Knowledge benchmark, Recursa lifts the held-out all-pass rate from **30.0% → 55.0%** with DeepSeek-V4.1-Flash and **35.0% → 46.0%** with Qwen3.8-Flash.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

Fine-tuning for domain adaptation requires gradient updates, a training stack, and the ability to modify model weights. In regulated domains, with closed-weight APIs, or when adaptation must be auditable, fine-tuning is impractical.

Recursa adapts a *system* around a frozen model: a shared knowledge repository that holds what the model has learned about the domain, and a loop that grows that repository by letting the model learn from its own task executions. Nothing about the model's parameters changes. The repository is a structured document that the model reads at inference and that human reviewers can inspect and edit.

---

## How it works

Four components:

1. **Knowledge repository.** A structured store with three entry types:
   - **Skills** — domain-specific procedures ("to query filing status, call X then Y").
   - **Application conditions** — when to apply each skill ("use this skill when the client mentions a jurisdiction").
   - **Reflections** — generalized lessons from past task failures.

2. **Bootstrapping.** Seed the repository from domain materials (manuals, policies, examples). This is a one-shot extraction pass, not a training run.

3. **Parallel learners.** For a batch of tasks, run $N$ workers in parallel. Each worker reads the current repository, executes the task, and produces (a) the answer and (b) a candidate improvement to the repository — a new skill, a sharpened application condition, or a reflection on a failure.

4. **Validation + merge.** A validator checks each candidate improvement against (a) the base policy already solving future tasks, (b) non-contradiction with existing entries, (c) demonstrable task-success delta on a validation slice. Accepted candidates are merged into the shared repository; rejected ones are discarded. The merged repository becomes input to the next iteration.

Recurse. Stop when validation-slice gains plateau.

---

## Why it matters

- **Gradient-free domain adaptation.** Works with API-only models; no access to weights, no training infrastructure.
- **Auditable.** Every behavioral change is a text entry in the repository that a human can read, edit, or revert.
- **Compatible with closed-weight frontier models.** The method applies to DeepSeek-V4.1-Flash, Qwen3.8-Flash, GPT-class, Claude-class — any model with sufficient context to read the repository.
- **Big empirical gains on held-out tasks.** +25 points (DeepSeek) and +11 points (Qwen) all-pass rate lift on Firm Knowledge, without training.
- **An alternative RL-free pathway.** Traditional post-training (SFT → RL) is the dominant pattern for capability lift. Recursa is in a different family — the model's capabilities are unchanged, but the system's output quality rises because the context it reads is better.

---

## Gotchas & tricks

- **Validator is load-bearing.** A permissive validator fills the repository with noise that drags inference quality down. A strict validator rejects useful improvements. Calibrate on a held-out slice before unleashing the loop.
- **Non-contradiction matters.** Two entries that each win in isolation can conflict when both apply — e.g. a skill that always calls tool X and another that always skips tool X. The validator must check pairwise consistency, not just marginal gain.
- **Repository size is an operational concern.** As it grows, context cost grows linearly. The paper doesn't give a concrete cap; in practice, structure the repository so inference reads only relevant chunks (retrieval + inclusion).
- **Doesn't help where the model lacks the underlying capability.** Recursa amplifies capabilities the model already has by giving it better instructions; it can't manufacture missing capabilities. Pair with fine-tuning or model switching where that gap exists.
- **Reflections drift toward platitudes.** Workers asked "what did we learn?" often produce vague generalizations. Constrain reflection generation to explicit failure/success contrasts.

---

## Sources

- Paper: *Recursa: Toward Recursively Self-Improving Foundation Model Systems* — Zhu, Lin, Shi, Lin, Zhou (Alibaba), 2026, arXiv 2611.04035 — primary source.
- Related: [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md) — a parallel line of work where external curated content shapes model behavior without weight updates.
