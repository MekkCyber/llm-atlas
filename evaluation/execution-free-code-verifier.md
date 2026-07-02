# Execution-Free Code Verifier
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard code-agent training relies on **execution-based** verification — spin up a per-repository Docker environment and run unit tests to score a candidate patch. An **execution-free** (or "environment-free") verifier judges patch correctness *without executing anything*: an LLM agent explores the repo, gathers evidence about the change, and outputs a verdict. Dockerless (2026) is the first such verifier that matches execution-based post-training on SWE-bench Verified while eliminating the container-orchestration cost entirely.

**Prereqs:** [README](README.md)
**Related:** [livecodebench](livecodebench.md) · [../post-training/rlvr](../post-training/rlvr.md) · [../post-training/rejection-sampling](../post-training/rejection-sampling.md) · [../agents/README](../agents/README.md)

---

## What it is

SWE-agent post-training pipelines (SFT trajectory filtering + RL reward) both need a *verifier*: given a candidate patch, is it correct? Two families:

- **Execution-based.** Per-repo Docker image, install deps, run unit tests. Ground-truth on repos where tests exist. Expensive to build and maintain — every repo needs its own reproducible environment.
- **Execution-free.** LLM agent reads the repo and the patch, follows definitions and callers, and reasons about whether the patch is semantically correct. No environment; no tests; no execution.

Execution-free verifiers were previously seen as strictly weaker (rewarding plausible-looking patches). Dockerless (Anon., 2026) provides the first evidence that a well-designed agentic verifier can *match* execution-based post-training in the SWE-agent setting.

## How it works

### The Dockerless agent verifier

For a candidate patch $p$ against repository $R$ and issue $I$:

1. **Read.** Ingest the issue text, the patch diff, and a summary of the repo layout.
2. **Explore.** Take agentic actions — open files, follow imports, search for callers of touched functions, inspect test files that touch the modified module.
3. **Judge.** Emit a verdict scalar (correctness score in [0, 1]) plus a short justification tied to the evidence collected.

The verifier is itself an LLM agent — a tool-using loop with a repository-navigation toolset. Because it *simulates* what a code reviewer would do, it is robust to patches that look plausible but break something upstream.

### Use in post-training

Dockerless plugs into the standard two-stage pipeline:

- **SFT trajectory filter.** Reject trajectories whose final patch scores below a threshold.
- **RL reward.** Use the verifier's scalar output as the RLVR reward. Combined with GRPO or PPO for the agent policy.

The whole pipeline becomes **environment-free**: no Docker builds, no test infrastructure, no per-repo CI images.

## Why it matters

- **+14.3 AUC over the strongest open-source execution-based verifier** on a verifier-quality benchmark.
- **SWE-bench Verified / Multilingual / Pro: 62.0 / 50.0 / 35.2** resolve rate, matching execution-based post-training and beating the Qwen3.5-9B baseline by 2.4 / 8.7 / 2.9 points.
- **Unblocks post-training on the long tail of repos.** Execution-based verification requires a working environment for every repo you train on. Execution-free verification requires only the ability to read source code. This scales to niche repos, obscure ecosystems, and languages without mature CI images.
- **Cheap to iterate.** A verifier bug is a prompt change, not a Docker image rebuild.

## Gotchas & tricks

- **Reward hacking.** Any LLM-judge reward is prone to being gamed by the policy. Ensemble with rule checks (patch touches only expected files; no obvious sink deletions) as a low-cost floor.
- **Coverage of the exploration tools matters.** If the verifier can't grep for callers or read imports, it will produce shallow judgments. Give it the same toolset a human reviewer would use.
- **Long-context repos.** Big repos may exceed the verifier's context window; long-context or agentic paging is required. Dockerless-style exploration is well-suited to this — it retrieves what it needs rather than reading the whole repo.
- **Doesn't replace tests at deployment time.** Execution-free verification is a *training* signal. Ship code with real tests; keep humans and CI in the loop for production.

## Sources

- Paper: *Dockerless: Environment-Free Program Verifier for Coding Agents* — Anonymous, 2026 — the execution-free verifier + fully environment-free post-training pipeline.
- Related: SWE-agent, SWE-bench Verified / Multilingual / Pro — the substrate benchmarks.
