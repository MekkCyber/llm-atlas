# RiskChainBench
*Depth — a two-stage benchmark for agent-driven investigation of obfuscated platform-abuse chains.*

**TL;DR:** RiskChainBench evaluates models on the actual platform-abuse pipeline: obfuscated messages (emojis, homophones, character decomposition, redundant symbols) that hide redirection to pornography, fraud, gambling, or illicit services. Stage 1 restores 3,600 synthetic obfuscated messages; Stage 2 investigates 600 human-labeled real web environments and files an evidence-grounded risk report. Entry Top-1 accuracy ranges 35.2%–95.2% across models; downstream web-decision accuracy 26.3%–62.8% — a wide spread that separates strong tool-using agents from weak ones.

**Prereqs:** [../safety/_attacks.md](../safety/_attacks.md), [../safety/character-encoding-obfuscation.md](../safety/character-encoding-obfuscation.md)
**Related:** [../safety/payload-splitting.md](../safety/payload-splitting.md), [../agents/agent-harness.md](../agents/agent-harness.md)

---

## What it is

A benchmark released with a sandbox environment and a full evaluation protocol. It measures two capabilities in sequence:

- **Restoration:** Given an obfuscated platform message, output the underlying redirection instruction. Obfuscations include emoji substitution, homophones, character decomposition (Chinese radicals), redundant-symbol padding.
- **Investigation:** Given the restored link (or an already-clean lead), the agent uses tools to visit and characterize the destination site, then produces an evidence-grounded risk report labeled against 600 human-annotated web environments.

Two headline metrics: **Entry Top-1** on restoration and **web-decision accuracy** on the investigation stage.

---

## How it works

```
obfuscated msg ─► restore ─► clean link ─┬─► agent tools ─► evidence
                                          │                        │
                                          └────────────────────────┴─► risk report
                                                                          │
                                                                    scored vs. 600 human labels
```

The sandbox packages a browser + tools so agents can be evaluated repeatably without live network variance. Restoration is scored against the ground-truth clean message; the investigation is scored end-to-end against human labels of the actual site's risk category.

---

## Why it matters

- **Fills a specific gap.** Between passive-refusal jailbreak benches and pure agent-navigation benches (WebArena, Terminal-Bench), no evaluation stressed *investigation-under-obfuscation*. RiskChainBench is that.
- **Uses real obfuscation traffic.** Not synthetic red-team prompts; the obfuscations are drawn from observed abuse patterns.
- **Discriminates well.** Wide spread on both stages means the benchmark is not yet saturated — useful for tracking progress.
- **Reproducible.** Sandbox + protocol released, so numbers are comparable across labs.

---

## Gotchas & tricks

- **Two-stage nature complicates comparisons.** Reporting only Stage 1 or only Stage 2 hides the pipeline. Report both.
- **Investigation stage stresses tool-use quality, not just reasoning.** A model with a weak harness may fail Stage 2 even with perfect Stage 1.
- **600 human-labeled web environments is finite.** Over-optimizing for the benchmark risks overfitting to those specific sites. Rotate held-out subsets when comparing across a long timeline.
- **Language/locale scope.** Obfuscation categories reflect the paper's source traffic; extension to other locales requires a new labeled set.

---

## Sources

- Paper: *RiskChainBench: A Benchmark for Obfuscated Platform Message Restoration and Evidence-Grounded Web Investigation* — Liu et al., 2026 — [arXiv:2609.16900](https://arxiv.org/abs/2609.16900).
