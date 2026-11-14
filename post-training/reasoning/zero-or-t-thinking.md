# Zero-or-T Thinking
*Depth — a binary gate on reasoning: either answer directly, or run a full T-token CoT, trained with a cost-aware DPO variant (CDPO).*

**TL;DR:** Always-on CoT wastes tokens on inputs where the model already knows the answer. Continuous adaptive-length CoT is expensive to train. "Zero-or-T thinking" makes the length choice **binary**: for each input, either skip CoT entirely (zero tokens) or run a fixed-length T-token reasoning trace. A small self-routing head attached to a frozen backbone decides per-input which branch to take, trained with **CDPO** — a cost-aware DPO variant that treats thinking as a priced option. Reported results: higher JevBench accuracy than a full-thinking baseline at much lower latency, and an **80.1% token-cost reduction on MiniWoB**.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [../dpo.md](../dpo.md)
**Related:** [length-penalty](length-penalty.md), [../grpo.md](../grpo.md)

---

## What it is

A hybrid reasoning policy with two modes:

- **Zero mode:** output the answer immediately, no CoT tokens generated.
- **T mode:** generate a full autoregressive CoT of fixed length T, then answer.

A **self-routing block** attached to the frozen backbone estimates the *value of thinking* $V_t(x)$ for input $x$ — the expected quality improvement from running T mode over Zero mode. The policy escalates to T mode iff $V_t(x) > c$, where $c$ is a per-request *price* (e.g., token budget in dollars).

The decision is binary, which is the design choice: it's easier to train a well-calibrated binary gate than a continuous length-chooser.

## How it works

### Architecture

- Frozen backbone LLM (no weight update during policy training).
- Lightweight self-routing head on top of the backbone's final hidden state (or an early-layer state, for cheap gating). Outputs a scalar $\hat{V}_t$.
- Decision: `if V_hat_t > price: run T-mode; else: run Zero-mode`.

### CDPO: cost-aware DPO

Standard DPO trains a policy from preference pairs $(y^+, y^-)$ without using an explicit reward model. CDPO extends it to **cost-aware preferences**:

For each prompt, generate both a Zero-mode answer $y_0$ and a T-mode answer $y_T$, score each with an outcome verifier (correctness), and form preferences:

- If $y_T$ is correct and $y_0$ is wrong: $y_T \succ_c y_0$ for all $c$.
- If both correct: $y_0 \succ_c y_T$ for $c > 0$ (Zero wins on cost).
- If both wrong: no preference signal.

The DPO log-ratio objective is augmented with a cost penalty on T-mode outputs:

$$
L_{\mathrm{CDPO}} = -\mathbb{E}[\log \sigma(\beta (\log \pi_\theta(y^+) - \log \pi_\theta(y^-)) - c \cdot |y^+|_\text{CoT})]
$$

The routing head learns to escalate exactly when the correctness gap exceeds the cost gap.

### Episodic variant

For multi-step agent tasks (MiniWoB), the paper extends CDPO with trajectory-level cost accumulation — the price on T-mode is paid *per step*, so the policy escalates sparingly across an episode, not independently per step.

## Why it matters

- 80% token savings at matched quality is a deployment-grade number for reasoning serving.
- Simpler than continuous-length adaptive CoT: one binary gate, one scalar price, no length-reward tuning.
- The CDPO objective generalizes to any "two modes with different cost and different quality" training problem — e.g., tool-use vs. no-tool, 7B vs. 70B cascading.

## Gotchas & tricks

- Price $c$ is a deployment knob. Training should sweep $c$ across a range and leave the router's calibration good across the range; otherwise the policy brittle-ly prefers one mode.
- The gate head's input needs to be early enough in the backbone to beat T-mode compute — gating on the final hidden state wastes a full forward pass before deciding not to think.
- Fixed T can be too short on hard problems and too long on easy ones. A natural extension is Zero/T₁/T₂ (two T values), fit into the same cost-aware DPO frame.

## Sources

- Paper: *The Value of Thinking: Learning When to Reason in Hybrid Decision Models* — Xu, Fang, Ammar, Luo, Wang, 2026 — https://arxiv.org/abs/2611.16961
