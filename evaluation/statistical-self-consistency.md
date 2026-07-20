# Statistical Self-Consistency
*Depth — reference-free evaluation via the law of total probability over verbalized subpopulations.*

**TL;DR:** If in-context learning is really conditional inference, LLM outputs should obey the **law of total probability**: prior-weighted subpopulation estimates should aggregate to the population marginal over any valid partition. **Partition, Prompt, Aggregate** uses binary trees to recursively partition, prompt an LLM per leaf, aggregate up, and compare against the direct population-level answer. Any mismatch is a self-consistency violation — **reference-free**.

**Prereqs:** [ifeval.md](ifeval.md)
**Related:** [mmlu.md](mmlu.md), [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)

---

## What it is

Most LLM evaluation protocols compare model outputs to a ground-truth reference (accuracy on labeled benchmarks). This has known problems: contamination, reference-quality ceiling, expensive labeling.

Statistical self-consistency evaluates a model *without any ground truth*, by checking whether its outputs are *internally consistent* under a basic probabilistic identity — the law of total probability:

$$P(y) = \sum_c P(y \mid c) \cdot P(c)$$

For any valid partition $\{c_1, \ldots, c_k\}$ of the population, prior-weighted conditional estimates should reconstruct the marginal. If the LLM's estimates violate this at scale, its outputs cannot be a valid conditional inference — evidence of miscalibration that requires no reference labels.

## How it works

### The protocol

1. **Choose a target quantity.** E.g. "what fraction of population $P$ agrees with statement $s$."
2. **Build a partition tree.** A binary tree that recursively splits $P$ into subpopulations along verbalizable attributes ("men vs women", "under 40 vs over 40", "US vs non-US", ...).
3. **Prompt per leaf.** For each leaf subpopulation, prompt the LLM with a verbalized description ("You are describing responses from US women under 40. What fraction agree with $s$?") and record the estimate.
4. **Aggregate up.** Combine the leaves via the law of total probability using prior weights $P(c)$ (either known or LLM-estimated).
5. **Compare.** The reconstructed marginal is compared against the direct population-level estimate ("What fraction of the general public agrees with $s$?"). Any gap is a violation.

Repeat across different tree structures (different partitioning orders) and different domains — persistent violations across tree structures indicate a genuine calibration failure, not tree-specific noise.

### The macro fallacy

The paper's most striking finding is what they call the **macro fallacy**: for many tasks, the *reconstructed* estimate — aggregated from fine-grained subpopulation queries — aligns *better with human reference data* than the direct population-level estimate. In other words, the model knows what subpopulations think but doesn't propagate it correctly into aggregate answers. This is not just a consistency failure; it's a practical prompting insight.

## Why it matters

- **Reference-free.** No labels needed — the criterion is self-consistency, not agreement with an oracle.
- **Unsaturated.** Frontier models violate self-consistency broadly; there's a lot of headroom for measuring model quality along this axis.
- **Robust across tree structures and tasks.** Widespread violations aren't artifacts of a specific partitioning.
- **A prompting recipe emerges.** For population-level estimates, ask subpopulations and aggregate — often better than direct queries.

## Gotchas & tricks

- **The partition must be verbalizable.** If leaf subpopulations can't be cleanly described in text, the LLM's per-leaf estimates aren't grounded and the test is meaningless.
- **Prior weights $P(c)$ matter.** Wrong priors bias the aggregate; verify priors are correct (from data) rather than LLM-estimated when possible.
- **Tree depth is a knob.** Deeper trees stress consistency more but each leaf becomes less well-defined. The paper varies depth to check that violations persist.
- **This tests self-consistency, not correctness.** A model can be perfectly self-consistent and still wrong; the criterion complements accuracy benchmarks, doesn't replace them.
- **Persona prompting reveals the strongest violations.** The macro-fallacy effect is most pronounced with persona-framed prompts, suggesting persona conditioning is a specific culprit.

## Sources

- Paper: *Partition, Prompt, Aggregate: Statistical Self-Consistency in Language Models* — Wolf, Kleine Buening, Krause, Mendler-Dünner — ETH Zurich / MPI-IS, 2026.
