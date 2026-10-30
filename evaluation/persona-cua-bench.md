# PersonaCUABench
*Depth — a computer-use agent benchmark that holds the task fixed and varies the user who asks.*

**TL;DR:** **PersonaCUABench** (Li et al., 17-university consortium, 2026) measures how robust a CUA is to **variation in the user who asks**, not variation in the task. 74 ground-truth tasks across 11 domains, each paraphrased by **44 personas in 4 demographic/cognitive groups**; a persona agent speaks in character and answers mid-run clarifications. Across 9 agent backbones and **3,330 total runs**, mean task completion drops **36%** vs. original (clean-spec) prompts. Most of the loss is from requirements the agent *engaged with* but still got wrong, plus requested stops the checker treats as failures. Even on fully-completed runs, top-rating user satisfaction is far rarer than on original prompts.

**Prereqs:** *(none)*
**Related:** [../agents/computer-use-agents.md](../agents/computer-use-agents.md), [_agent-benchmarks.md](_agent-benchmarks.md), [cli-agent-bench.md](cli-agent-bench.md)

---

## What it is

A computer-use-agent (CUA) benchmark structured around a controlled variable:

- 74 ground-truth tasks across 11 domains (file management, web navigation, email composition, data analysis, …).
- 44 personas across 4 groups (demographic, cognitive-style, communication-style, domain-knowledge-level).
- Every task is paraphrased by every persona ⇒ 74 × 44 = **3,256 persona-task cells**.
- Each cell runs with a scripted persona agent that paraphrases the instruction and answers clarifications in character.

The result for each (agent, task, persona) is a tuple:

```
task_completion:    did the final state match ground truth?
user_satisfaction:  what rating does the persona agent give?
engagement_quality: did the agent engage with requirements it still got wrong?
```

---

## How it works

### Persona generation

Personas are LLM-prompted characters with structured attributes:

- Age bracket, native language, communication directness, patience, domain expertise.
- A canonical speech register.
- A pre-committed "satisfaction rubric" (what will make them give top rating).

The persona generator produces a template; the persona agent sticks to it across the whole run.

### Instruction paraphrasing

For each original task spec `t_0`:

```
for persona p:
    t_p = persona_agent[p].rewrite(t_0)
```

The paraphrase preserves the goal but reframes it in the persona's voice. Some personas are terse; others verbose; some leave key information implicit.

### Clarification turns

During the agent's run, if the agent asks for clarification, the persona agent answers in character — sometimes providing the info, sometimes refusing, sometimes giving partial info that requires further asking.

### Grading

```
task_completion:    automated outcome checker (same as original task)
user_satisfaction:  5-point scale from persona agent after the run
loss_decomposition: engaged-but-wrong, requested-stop, unaddressed-requirement
```

The loss decomposition is the key diagnostic — it tells you *why* a run failed.

---

## Why it matters

- **Operationalizes a long-known failure mode.** Users have always said "the agent doesn't understand me." PersonaCUABench measures that explicitly.
- **The 36% drop is enormous.** For a benchmark varying only the user voice (same task, same ground truth), losing more than a third of task completion means current CUAs are brittle to real human variability.
- **Completion ≠ satisfaction.** Even for runs that succeed, top satisfaction is far rarer than on original prompts. "Did the task" is not "served the user."
- **Loss decomposition is actionable.** "Engaged-but-wrong" points to understanding failures; "requested-stop" points to over-conservatism; "unaddressed-requirement" points to attention failures. Each has a different fix.
- **Reframes CUA evaluation around *user intent fidelity*, not task completion.**

---

## Gotchas & tricks

- **Persona agent biases.** The 44 personas reflect the paper's prompting choices. A benchmark with 44 different-but-realistic personas would give different numbers; the direction (drop vs original) should be stable.
- **Satisfaction is persona-defined.** The persona has an internal satisfaction rubric; agents cannot know it exactly. This is intended — real users have hidden satisfaction criteria — but it bounds reproducibility.
- **Task set is English-centric.** 11 domains are specified in English. Multilingual persona variation is a natural extension not covered.
- **Benchmark saturates slower than outcome-only.** Expect CUA improvements to show up here before they show up on simpler OSWorld-style tasks, since adding user variation adds a new axis of difficulty.
- **Not a replacement for outcome benchmarks.** Combine with OSWorld / Windows Agent Arena for a multi-axis picture.

---

## Sources

- Paper: *PersonaCUABench: Benchmarking Computer-Use Agents Serving Diverse Personas* — Li et al., 17-university consortium led by UT Dallas / USC / CMU, 2026, [arXiv 2610.22990](https://arxiv.org/abs/2610.22990).
- Related: OSWorld (2024), Windows Agent Arena (2024) — outcome-only CUA benchmarks.
- Related: [../agents/computer-use-agents.md](../agents/computer-use-agents.md), [_agent-benchmarks.md](_agent-benchmarks.md).
