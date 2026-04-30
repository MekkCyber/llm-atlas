# Llama Guard

*Depth — a fine-tuned Llama-3 model that classifies prompts and outputs against a safety taxonomy.*

**TL;DR:** A **fine-tuned Llama 3 8B** used as a safety classifier in front of / behind the main model. Inputs: a prompt and/or a response; outputs: SAFE/UNSAFE label + category of hazard (if UNSAFE). Trained on the **MLCommons 13-category hazard taxonomy** + Code Interpreter Abuse. Released by Meta alongside Llama 3. **Multilingual** (English + others). Int8-quantized variant available (>40% size reduction, negligible quality loss). Median **~50–86% reduction in Violation Rate** across languages at the cost of +26–102% False Refusal Rate. Similar tools: **Prompt Guard** (86M mDeBERTa for jailbreak detection), **Code Shield** (static insecure-code analysis).

**Prereqs:** [mismatched-generalization](mismatched-generalization.md), [competing-objectives](competing-objectives.md)
**Related:** [prompt-guard](prompt-guard.md) · [code-shield](code-shield.md) · [rainbow-teaming](rainbow-teaming.md) · [uplift-evaluation](uplift-evaluation.md)

---

## What it is

A sidecar safety classifier, architecturally a fine-tuned **Llama 3 8B**. Designed to wrap around the main model at deploy time:

- **Input guard**: user prompt → Llama Guard → SAFE / UNSAFE? If UNSAFE, block.
- **Output guard**: model response → Llama Guard → SAFE / UNSAFE? If UNSAFE, regenerate or block.
- Optimized for **tool-call contexts** (search, code interpreter) where safety matters most.

Meta's Llama Guard has evolved through several versions (Llama Guard 1 → 2 → 3). Version 3 ships with Llama 3.1.

---

## How it works

### Taxonomy — the 13 MLCommons hazard categories + 1

1. Child Sexual Exploitation (CSE)
2. Defamation
3. Elections
4. Hate
5. Indiscriminate Weapons (bio, chem, nuclear, radiological)
6. Intellectual Property
7. Non-Violent Crimes
8. Privacy
9. Sex-Related Crimes
10. Sexual Content
11. Specialized Advice (medical, legal, financial)
12. Suicide & Self-Harm
13. Violent Crimes
+ **Code Interpreter Abuse** (Llama Guard 3 specific)

### Input/output format

Input format is a prompt template:

```
Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
O1: Violent Crimes.
O2: Non-Violent Crimes.
...
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: {user_message}
Agent: {agent_message}
<END CONVERSATION>

Provide your safety assessment for Agent in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.
```

Model output:
```
unsafe
O5, O8
```

(Unsafe in categories O5 = Indiscriminate Weapons and O8 = Privacy.)

### Fine-tuning

Llama Guard 3 starts from Llama 3 8B and is fine-tuned on:
- Prompts labeled for each of the 13+ categories.
- Multilingual examples.
- Tool-context examples (search query + result pairs, code-interpreter inputs).
- Adversarial examples (jailbreak prompts that should still be classified correctly).

Exact training data size is not fully disclosed; the system card describes the taxonomy and evaluation but not the training set size.

### Int8-quantized variant

A compressed Llama Guard 3 quantized to INT8 weights. >40% size reduction. Quality delta on eval: precision 0.947 → 0.947 English (unchanged), recall 0.931 → 0.925 (small drop). Pareto-dominant for deployment.

### Deployment pattern

```
user_input
    ↓
Llama Guard (input check) → UNSAFE? → block with refusal
    ↓ (if safe)
Llama 3 Instruct → response
    ↓
Llama Guard (output check) → UNSAFE? → regenerate / block
    ↓ (if safe)
Return response to user
```

Two-gate pattern: check both input and output. Input check catches attempted harmful queries; output check catches model-generated harmful content even on seemingly-safe inputs.

---

## Results

From Llama 3 paper Table 25 (main safety eval):
- **~50–86% median reduction in Violation Rate** across languages.
- **+26–102% increase in False Refusal Rate** (the model refuses safe queries it shouldn't).
- Net: Pareto-safer than competing API systems on multilingual safety benchmarks (Figure 19).

The false-refusal cost is real. Llama Guard is conservative; deployers should benchmark on their own risk-tolerance curve.

---

## Why it matters

- **Standard open safety sidecar.** Llama Guard is what most open-stack deployments use. Cheaper and smaller than having the main model refuse everything.
- **Decouples safety from the main model.** The main Llama 3 can be tuned for capability; Llama Guard handles the safety gate. Easier to iterate on each.
- **Multilingual by default.** Unlike many safety classifiers trained only on English, Llama Guard 3 handles multiple languages — matches Llama 3's multilingual deployment.
- **Fine-tuned model beats zero-shot.** A 8B specialized safety model is more accurate and faster than zero-shot asking Llama 3 70B "is this safe?"

---

## Gotchas & tricks

- **Conservative by default.** False refusal rate is the trade-off. Adjust the threshold (or fine-tune further) for your risk tolerance.
- **Category overlap.** Some prompts violate multiple categories. Llama Guard returns the comma-separated list; deployers should handle multi-category responses.
- **Jailbreaks that evade Llama Guard exist.** Llama Guard is not bulletproof. Use in layered defense (also use [prompt-guard](prompt-guard.md) for jailbreak-specific detection and [code-shield](code-shield.md) for code).
- **Not a reward model.** Llama Guard is binary safe/unsafe; not a continuous score. Don't use for PPO.
- **Latency.** An 8B model as a gate adds latency. The int8 variant helps. For high-throughput serving, batch Llama Guard calls.
- **Mass-assignment vulnerability.** If you pass arbitrary user prompts into Llama Guard's template unescaped, a user could inject a fake `<END CONVERSATION>` and then instructions. Use proper escaping.
- **Don't show Llama Guard's output to the user.** The unsafe categories reveal what the user tried to do; informing them helps probe the system. Return a generic refusal.
- **Tool-call context.** Llama Guard 3 is specifically tuned for search-query and code-interpreter inputs (these are the highest-risk contexts in practice). Generic chat safety is a secondary focus.
- **Bias concerns.** Safety classifiers trained on Western data tend to over-flag non-Western cultural content. Multilingual Llama Guard 3 partially mitigates but not fully.
- **Deployment is layered.** Llama Guard is one layer; also consider input sanitization, rate limits, RAG-scope restrictions, user authentication, deployed-context limits.
- **Llama Guard 1 / 2 / 3 are different models.** 3 is current with Llama 3.1. Don't deploy older versions.

---

## Sources

- Paper: *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations* — Inan et al., Meta, 2023, arXiv 2312.06674 — the original Llama Guard.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 5.4.7 — Llama Guard 3 details.
- Hugging Face: `meta-llama/Llama-Guard-3-8B` — the model card.
- MLCommons hazard taxonomy: https://mlcommons.org/ — the 13-category taxonomy.
