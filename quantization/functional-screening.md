# Functional Screening for Quantized LLM Artifacts
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An empirical census of public LLM quantization artifacts (Ollama official tags + popular HF uploads) finds seven Ollama-official tags that produce non-functional output — all are low-bit quantizations of small models. Rebuilding them from original weights without an importance matrix (iMatrix) reproduces the failure; adding an iMatrix restores function. The paper argues public registries should *functionally screen* low-bit tags before publication and gives a lightweight test.

**Prereqs:** [README.md](README.md), [_number-formats.md](_number-formats.md)
**Related:** [fp8.md](fp8.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Low-bit (<4 bit, often GGUF-style) quantizations of small models can land in a state where the model loads but generates garbled or empty text. The failure isn't file corruption — it reproduces from the original weights under the same recipe. The ingredient missing in the broken cases is calibration via an importance matrix (iMatrix), which steers the quantizer away from destroying columns the model actually relies on.

## How it works

**Screening protocol.**
1. Load the candidate quantized file in its canonical inference stack (llama.cpp / Ollama / vLLM).
2. Run a small bank of generic prompts (completions, instructions, Q&A).
3. Score outputs with a *token-coherence* threshold — rolling perplexity, repetition rate, or UTF-8 validity.
4. Flag the artifact if the score drops below the threshold on more than a small fraction of prompts.

**Repair.** For a failing low-bit tag:
1. Compute an importance matrix (iMatrix) from a short calibration corpus (a few hundred samples typical).
2. Re-quantize with the iMatrix supplied to the quantizer.
3. Rerun the screen; function is typically restored.

**Census findings.** Seven Ollama-official tags fail the screen; all are low-bit quantizations of small models (small models have less redundant capacity to absorb quantization error). Rebuilding from scratch without iMatrix reproduces every failure. Rebuilding with iMatrix restores function for the measured subset.

## Why it matters

Public registries ship these artifacts to end users — edge deployments, local chatbots, on-device assistants — exactly where silent breakage is hardest to detect. A trivial pre-publish screen catches all measured cases. The methodology is portable: any registry (HF, Ollama, Together, custom) can add it to its upload pipeline. More broadly, this is the first systematic census of "shipped-but-broken" quantization artifacts; the pattern — "low-bit + small model + no iMatrix" — is a sharp predictor.

## Gotchas & tricks

- **Small models fail more often.** They have less redundant capacity to absorb quantization error; low-bit tags of 1–3B models are the high-risk bucket.
- **iMatrix calibration data should match deployment distribution.** Wiki-text iMatrix calibration can under-cover code or multilingual deployment; use a representative corpus.
- **Screen on canonical runner.** A tag can fail in llama.cpp but succeed in a custom runner that silently promotes precision; test on the runner users actually use.
- **Token-coherence thresholds need per-model baselines.** A 1B model's baseline perplexity is high; use relative drops vs. the FP16 reference, not absolute cutoffs.
- **Not a quantization algorithm.** This is a *QA process*; GPTQ/AWQ/GGUF produce the artifacts, iMatrix fixes a subclass of failures, and this screen catches whatever slips through.

## Sources

- Paper: *Broken on Arrival: Non-Functional Quantized LLM Artifacts in Public Model Registries and How to Catch Them* — (byline missing on page), 2026 — [arXiv:2609.05881](https://arxiv.org/abs/2609.05881).
- Reference: *llama.cpp* GGUF toolchain and the iMatrix / importance-matrix calibration flow.
- Reference: *GPTQ* and *AWQ* — the broader PTQ family these artifacts come from.
