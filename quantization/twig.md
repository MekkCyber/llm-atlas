# TWIG — Tree-structured Weights with Inherited Gates
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Represent every neural network weight as the **terminus of a small path of discrete decisions** — a sign, a power-of-two shift, a zero gate, or a table index. Weights that are related share the upper part of their decision paths. This single structure subsumes pruning, quantization, low-rank factorization, codebooks, and weight sharing: each becomes an *edit* to the same object, selectable by one optimizer. Early results on five LLM weight matrices: **9–15% fewer additions than GPTQ** at 3–4 bits on typical matrices, **20–55% fewer** on outlier-heavy matrices.

**Prereqs:** [_number-formats.md](_number-formats.md), [fp8.md](fp8.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

Weight compression techniques are usually studied in isolation: GPTQ for quantization, structured sparsity for pruning, SVD for low-rank, product quantization for codebooks. TWIG re-casts all of them as different *settings* of one representation — a decision tree whose leaves are the network's weights.

## How it works

- Every weight $w_{ij}$ is written as a composition of **discrete decisions**: a sign bit, a power-of-two shift (equivalent to a quantization level), an optional zero gate (equivalent to a pruning mask), and optionally a table index (equivalent to a codebook lookup).
- Groups of related weights share the **upper levels** of their paths — "inherited gates" — so one decision determines a whole subtree at once. This is how low-rank factorization and weight sharing emerge from the same representation.
- Because all decisions are discrete, choosing them is a **combinatorial optimization problem**. The paper argues it is equivalent to finding the ground state of a higher-order Boltzmann machine, which classical annealers and quantum optimizers natively target.
- Decoding a weight requires only wires and adders when the decisions are powers of two — no multipliers — which maps cleanly onto fixed-weight ASICs/FPGAs.

## Why it matters

- **Unifies five techniques into one search problem.** You can smoothly trade pruning intensity against bit-width against codebook granularity by changing where the shared gates sit, instead of running five different pipelines.
- **Inference-cost framing, not just bit-width.** The reported metric is additions-per-forward-pass, which is what hardware actually pays. At equal layer error, TWIG beats GPTQ by 9–15% on typical matrices and 20–55% on outlier-heavy ones.
- **Hardware co-design hook.** Because the representation is add-only, it suggests ASIC/FPGA accelerators that skip multiplication entirely for inference.
- **Early evidence, not a sweep.** The paper reports five matrices from one LLM; the authors frame it as first evidence rather than a finished benchmark. Expect follow-up work on end-to-end quality and cross-architecture generality.

## Gotchas & tricks

- **The optimizer is the hard part.** Finding good discrete decisions is NP-hard in general; the Boltzmann-machine framing suggests but does not deliver a scalable solver. Practical implementations likely use coordinate descent, simulated annealing, or hybrid continuous relaxations.
- **Outlier-friendly, outlier-dependent.** The big gains come on matrices with heavy outliers (where GPTQ wastes bits). On well-behaved matrices the win is single-digit; TWIG is not strictly better everywhere.
- **End-to-end accuracy is not yet reported.** Layer-level error at matched additions is the reported metric; end-to-end perplexity / downstream benchmarks are the obvious next test.
- **Not a drop-in for existing kernels.** Standard GEMM kernels expect multipliers; TWIG inference needs custom add-tree kernels to realize the hardware win.

## Sources

- Paper: *Every Weight Is a Path of Shared Discrete Decisions* — Lefkimmiatis (LEAD Lab, ITMO University), 2026 — https://arxiv.org/abs/2610.31931
