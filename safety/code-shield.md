# Code Shield

*Depth — static analysis for LLM-generated code to detect insecure patterns.*

**TL;DR:** Meta's Code Shield is an **Insecure Code Detector** — static analysis (not an LLM) that scans LLM-generated code for known insecure patterns across **7 languages** at inference time. Part of Llama 3's safety toolkit (paper Sec. 5.4.7). Not a classifier in the ML sense; it's rule-based security analysis similar to classical static analyzers (Bandit, Semgrep). Deployable as a hook in code-generation pipelines to flag or block known-bad outputs.

**Prereqs:** *(none)*
**Related:** [llama-guard](llama-guard.md) · [prompt-guard](prompt-guard.md)

---

## What it is

A **static analysis tool** (not an LLM) that runs on code emitted by an LLM. Classical program-analysis techniques — pattern matching, AST inspection, data-flow tracking — applied to detect insecure code constructs.

Covers **7 languages**: Python, C, C++, Java, JavaScript, Rust, and one more (the paper doesn't fully enumerate; these are the common ones Meta explicitly tests).

Detects standard insecurity classes:

- **SQL injection** — concatenation of user input into SQL strings without parameterization.
- **Command injection** — `os.system(user_input)`, shell-concat.
- **Path traversal** — unfiltered path operations.
- **Hardcoded credentials** — API keys, passwords in source.
- **Insecure cryptographic primitives** — MD5, SHA1 for security, ECB mode, weak random.
- **Unsafe deserialization** — `pickle.loads(untrusted_input)`, YAML unsafe_load.
- **Memory-safety issues** (in C/C++) — unbounded `strcpy`, use-after-free patterns.
- **XSS patterns** — unescaped HTML output.

### Deployment pattern

```
user_prompt → Llama 3 → generated code
                              ↓
                         Code Shield
                              ↓
                   detected insecure pattern?
                              ↓
               YES → warn user / strip / block → OR regenerate
               NO  → return to user
```

At inference time, before surfacing code to the user or executing it in a sandbox.

### Why not an LLM classifier

Meta's choice of static analysis over an LLM-based classifier:
- **Deterministic** — same input → same output. No false-negative by hallucination.
- **Fast** — ms-range inference, no GPU.
- **Explainable** — the rule that matched is human-readable, pointing to the specific line.
- **Covers well-defined patterns** — decades of security tooling has catalogued these.

LLM-based analysis would catch a wider range of subtle issues but at far higher compute, with non-determinism, and less explainability. Meta uses both: static analysis for known patterns (cheap, reliable) and LLM-based reasoning (slower, for less-codified issues).

---

## Why it matters

- **Layered defense.** Even if the LLM generates insecure code (a real risk, especially for less common languages or obscure APIs), Code Shield catches the pattern before deployment.
- **CyberSecEval alignment.** Llama 3's own safety-eval (CyberSecEval, paper Sec. 5.4.5) shows LLMs do generate insecure code at non-trivial rates (10.4% for 405B on code-interpreter abuse queries). Code Shield is one tool to mitigate this in production.
- **Reusable for any LLM.** Not specific to Llama 3. Drop into any code-generation pipeline.
- **Production-ready.** Static analysis has decades of tool maturity; Code Shield inherits that.

---

## Gotchas & tricks

- **False positives are common.** Any static analyzer for security has a false-positive rate. Tune to your risk tolerance.
- **Doesn't catch novel vulnerabilities.** Zero-day patterns not in the rule database slip through.
- **Doesn't reason about code semantics deeply.** A careful attacker can write code that looks innocent but is malicious (obfuscated code, side-channels, timing attacks) — static analysis misses these.
- **Language coverage.** 7 languages is a lot by ML standards; tiny by the full space of languages. Go, Ruby, PHP, Kotlin, Swift, TypeScript are covered in some variants but check for your use case.
- **Rule maintenance.** New vulnerability patterns require updating the rule database. Meta periodically publishes updates.
- **Integration is up to the deployer.** Code Shield is a CLI/library, not a default part of the Llama serving stack. Hook it in yourself.
- **Doesn't replace code review.** A tool, not a ceiling. For production code generation, humans still review. Code Shield is the first-pass filter.
- **Composable with runtime sandboxing.** Pair with containerized execution (e.g., the Kimi k1.5 code sandbox) — static analysis catches patterns; sandboxing limits blast radius if patterns miss.
- **No published training data.** Code Shield is rule-based, so there's no training data. Detection quality depends on the rule database's coverage.

---

## Sources

- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 5.4.7.
- Paper: *CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models* — Bhatt et al., Meta, 2024, arXiv 2404.13161 — measurement of insecure-code-generation rates.
- Repo: Meta's Purple Llama project — https://github.com/meta-llama/PurpleLlama — includes CyberSecEval and related tooling.
- Related tools: Bandit (Python), Semgrep (multi-language), CodeQL (GitHub) — classical static analyzers Code Shield resembles.
