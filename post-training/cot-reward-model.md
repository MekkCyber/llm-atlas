# Chain-of-Thought Reward Model (CoT RM)
*Depth — a generative reward model that emits a reasoning trace before its judgment, improving verifier accuracy over classic value-head RMs.*

**TL;DR:** Classic reward models for LLM RL have a scalar head on top of a backbone — one forward pass, one number out. A **CoT RM** instead **generates a step-by-step reasoning trace** evaluating the response, then emits a structured judgment (e.g., JSON with a correctness boolean). Same data (question, reference answer, candidate response); different output format. Kimi k1.5 reports **spot-check accuracy jumping from ~84.4% (classic value-head RM) to ~98.5% (CoT RM)** on math verification — with the same ~800k training examples. A specific instance of the broader **LLM-as-judge / generative reward model** family. Slower per call, but high enough accuracy gain that the per-call cost is worth paying in RL.

**Prereqs:** [_rewards](_rewards.md), [rlvr](rlvr.md), [orm](reasoning/orm.md)
**Related:** [grpo](grpo.md) · [online-policy-mirror-descent](reasoning/online-policy-mirror-descent.md) · [rejection-sampling](rejection-sampling.md) · [kimi-k1-5 case study](../case-studies/kimi-k1-5.md)

---

## What it is

A learned reward model whose output is **a generated reasoning trace plus a structured verdict**, instead of a scalar head reading off a pooled representation. Specifically (Kimi k1.5 Sec. 2.3.5):

- **Input**: `(question, reference answer, candidate response)`.
- **Output**: step-by-step reasoning about whether the candidate matches the reference, terminating in a **JSON judgment** like `{"correct": true}` or `{"correct": false}`.
- **Training data**: same as a classic RM — `(x, y_ref, y_cand, label)` triples, where the label is 0/1 correctness (from rule verifier ground truth).

The difference from a classic value-head RM is structural: the value-head RM reads the candidate and emits a scalar; the CoT RM **reasons about** the candidate first and emits a judgment. Both are trained on the same labels; the CoT RM's intermediate reasoning is either generated from the underlying LLM's own knowledge (zero-shot CoT) or itself trained on CoT-annotated data.

This is a specific instance of the general "LLM-as-judge" / "generative reward model" pattern — not novel to Kimi as a concept, but the paper's 84% → 98% accuracy gap is a concrete and useful data point for why the pattern is adopted.

---

## How it works

### Training data

Kimi reports **~800k training examples** for both the classic RM and the CoT RM (same data; different output shape).

- **Classic RM**: emits a scalar (value head on backbone).
- **CoT RM**: emits reasoning + JSON judgment. Trained on a variant of the same data where the targets include the reasoning trace (either human-authored or distilled from a stronger model).

### Inference

For each candidate response during RL:

```
prompt = format(question, reference_answer, candidate_response)
output = LLM_reward_model.generate(prompt, max_tokens=...)    # emits CoT + JSON
judgment = parse_json(output)
reward = 1 if judgment['correct'] else 0
```

The CoT output is discarded after parsing; only the boolean in the JSON flows into the RL update.

### Why the structural change helps

Value-head RMs compress the entire judgment into a single scalar at pooling time. For math, this means the RM must *silently* determine: "is `2/3` equivalent to `0.666...`?", "does this solution reach the same final answer as the reference?", "did the candidate go through a correct derivation or did it just guess?" — all inside a single forward pass, collapsed to one number.

A CoT RM gets to **do** the verification, step by step, and the generated trace is the mechanism for multi-step judgment. The underlying LLM's reasoning capability is put to work *as a verifier*.

This is the same reason CoT prompting helps in the first place — chains of reasoning outperform single-shot inference for problems that decompose. Verification decomposes.

### Reported accuracy gap

From Kimi k1.5 Sec. 2.3.5, on spot-check accuracy against a held-out set:

- Classic value-head RM: **~84.4%**
- CoT RM: **~98.5%**

14 percentage points of verifier accuracy, with the same training data. This is a big gap — and it matters for RL because RL's ceiling is set by the verifier (reward-function ceiling is model-output ceiling is reward-function ceiling). A verifier that is 98.5% accurate is nearly as clean a signal as a rule-based checker; an 84.4% verifier leaves 15% of reward signal miscalibrated.

### Kimi's choice: use the CoT RM during RL

Despite the per-call cost (generating CoT + JSON is much more compute than scalar inference), Kimi uses the **CoT RM during RL** (Sec. 2.3.5). For math RLVR with unverifiable intermediate steps, the accuracy gain outweighs the compute cost.

---

## Why it matters

- **Named bridge between rule-based and fully-learned verifiers.** A rule verifier is cheap and exact but limited to verifiable domains. A value-head RM is learnable and generalizes but noisy (15% noise ≈ meaningful reward hacking surface). A CoT RM is learnable, generalizes, AND nearly as accurate as a rule verifier — inside the domain it was trained on.
- **Empirical case for LLM-as-judge with explicit reasoning.** A lot of production "LLM-as-judge" setups are zero-shot CoT from a strong model (GPT-4, Claude) reasoning about a response. Kimi shows a trained specialist CoT RM can push verifier accuracy above zero-shot judges.
- **Concrete number to cite.** The 84.4% → 98.5% gap is a rare reported benchmark on RM quality. Most RM papers report downstream RL improvements (which mix many variables) rather than standalone verifier accuracy.
- **Unlocks RL on fuzzy-verifiable tasks.** Pure rule verifiers work on answer-matching math and pass/fail code. CoT RMs extend RLVR to tasks where correctness is judgeable (by reasoning about the response) but not mechanically checkable.

---

## Gotchas & tricks

- **Per-call cost is significant.** A value-head RM is one forward pass over the response. A CoT RM generates reasoning — often 200–1000 tokens — then the verdict. At RL scale with thousands of rollouts per step, this multiplies the verifier cost by 200×–1000×. Kimi's trade-off was: 14-point accuracy gain is worth it.
- **Latency in the RL loop.** Slow verifiers stall rollout throughput. Efficient implementations batch CoT RM calls, use vLLM for the verifier, and overlap with rollout generation.
- **CoT RM can itself be reward-hacked.** If the underlying LLM has biases ("longer responses look more thoughtful"), the CoT RM inherits them. Spot-check accuracy alone doesn't detect this; you need adversarial / OOD evals.
- **JSON parsing is a failure mode.** If the CoT RM emits malformed JSON ~0.5% of the time, RL sees 0.5% of rewards as "undefined." Robust parsers default to a conservative verdict (usually 0) and log the failure.
- **Training data quality matters more than architecture.** The 84% → 98% gap is about format, but also about training signal — the CoT RM was trained on CoT-annotated data. Poor CoT annotations yield poor CoT RMs, potentially worse than a value-head RM with good scalar labels.
- **The CoT is ignored at inference — but it's not wasted.** The discarded CoT can be logged for debugging: manually inspecting a few "verifier said wrong but rule says right" cases shows what the verifier is confused about, and is actionable training signal for the next iteration.
- **Domain mismatch risk.** A CoT RM trained on math might score well in-domain and poorly OOD (e.g., science reasoning, code). Unlike rule verifiers (which don't generalize at all), CoT RMs generalize *some*, but their generalization is bounded by their training distribution.
- **Rule verifier > CoT RM where both apply.** Where a rule verifier is available (answer match on integers, test case execution), it's essentially free and essentially perfect. Use the rule verifier and save the CoT RM for fuzzy cases.
- **Composable with rule verifiers.** Stack: use rule verifier where possible; fall back to CoT RM on prompts where the rule verifier can't judge. Kimi does this for math (rule for clean answer match; CoT RM for reasoning-centric problems where the answer needs interpretation).
- **Not a standalone reasoning capability.** CoT RM is a verifier, not a solver. Don't expect a CoT RM trained on correctness judgments to be useful for generating solutions.

---

## Relation to the broader RM landscape

| Reward type | Cost per call | Accuracy | Hackability | Generalization |
|---|---|---|---|---|
| Rule verifier | ~free | Perfect (in-domain) | Minimal | None (narrow) |
| Value-head RM (classic ORM / preference RM) | 1 fwd pass | ~70–90% | High | Medium |
| **CoT RM** | **CoT-length fwd pass** | **~95–99%** | **Medium** | **Medium-high** |
| Process reward model (PRM) | N forward passes (one per step) | Varies, high for reranking | High | Low |
| Human judge | Minutes | Depends | N/A | High |

CoT RM sits in the "trained learned RM with high accuracy, higher compute per call" cell. The main alternative with similar accuracy is zero-shot LLM-as-judge with a strong off-the-shelf model (GPT-4, Claude) — which is typically slower and opaque to training-time iteration.

---

## Sources

- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI (Kimi Team), 2025, arXiv 2501.12599 — introduces the CoT RM and reports the 84.4% → 98.5% gap (Sec. 2.3.5 "Reward Modeling for Math").
- Paper: *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — Zheng et al., 2023, arXiv 2306.05685 — foundational LLM-as-judge study; the CoT RM is a trained specialist version of this pattern.
- Related: [orm](reasoning/orm.md) — outcome reward model, the classic value-head alternative.
- Related: *Generative Verifiers: Reward Modeling as Next-Token Prediction* — Zhang et al., 2024, arXiv 2408.15240 — concurrent study of generative RMs with explicit reasoning; primary source for generative reward modeling as a named pattern.
