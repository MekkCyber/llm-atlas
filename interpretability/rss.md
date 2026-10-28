# Representation Redistribution Score (RSS)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Semantics-preserving prompt rewrites (paraphrases, format changes) don't just add noise to SAE activations — they systematically **redistribute** which features fire. RSS is a per-feature statistic that keeps a feature only if its activation shifts under rewriting in a **consistent direction** across many paraphrases. Across 27 pretrained SAEs from 8 LMs, RSS-selected features (median ~20) combined with a frozen linear safety readout rank recurrent failures with **AUROC 0.935–0.958**, within ~1 point of the full-feature AUROC 0.964–0.965.

**Prereqs:** [sae.md](./sae.md)
**Related:** [../safety/_jailbreaks.md](../safety/_jailbreaks.md), [../safety/mismatched-generalization.md](../safety/mismatched-generalization.md)

---

## What it is

Most SAE work scores features in isolation (max activation, logit effect). RSS is **relational**: it compares the same feature's activation on paired, meaning-preserving prompt variants.

Given a prompt $p$, a set of semantics-preserving rewrites $\{p^{(1)},\dots,p^{(k)}\}$, and feature activations $f_j(p)$ from an SAE, RSS computes the directional consistency of the shift $\Delta_j^{(i)} = f_j(p^{(i)}) - f_j(p)$ across $i$. Features whose shifts flip sign across rewrites are noise; features whose shifts agree in direction are reproducible — RSS keeps those.

## How it works

1. **Build paraphrase pools.** For each held-out prompt, generate several semantics-preserving rewrites (paraphrase, reformat, re-order clauses).
2. **Collect per-feature shift vectors.** For every SAE feature $j$, record its shifts $\Delta_j^{(i)}$ across the rewrite set.
3. **Score directional consistency.** RSS summarizes how much the shifts agree in sign (and optionally magnitude) across the paraphrase pool. Features with high consistency are kept.
4. **Downstream use.** Combine the RSS feature subset with the **clean decision margin** of a frozen linear safety readout to rank recurrent failures. A small set (~20 features) suffices.

The paper validates RSS by selecting on one pool of rewrites and testing on a disjoint pool, controlling for selection-set overfitting.

## Why it matters

Jailbreak and safety-failure modes commonly travel through paraphrase (mismatched generalization). Identifying *which features move together* under paraphrase gives a cheap, model-agnostic handle on exactly those failure modes. The ~20-feature subset is also small enough to inspect manually and build auditable classifiers from.

## Gotchas & tricks

- **Depends on the paraphrase distribution.** RSS is only as good as the rewrite pool: narrow pools (e.g. only lowercase variants) will surface narrower features.
- **Not a causal test.** RSS flags features whose activations shift together; a causal patching experiment is still needed to show those features drive behavior.
- **Readout stays frozen.** The headline numbers combine RSS features with a *fixed* linear safety classifier — not a re-trained one. If the readout is weak, RSS cannot rescue it.
- **Compact but not interpretable by itself.** The 20-feature set captures the signal; interpreting what each feature means still requires standard SAE labeling.

## Sources

- Paper: *Representation Redistribution in Sparse Autoencoders* — Haotian Ma, University of Wisconsin–Madison, 2026 — [arXiv:2610.21240](https://arxiv.org/abs/2610.21240).
