# Cold-Start Safety Gap

*Depth — tool-calling LLM agents are most vulnerable at the very start of a session and become substantially safer after a few benign agentic interactions.*

**TL;DR:** Safety in a tool-calling LLM agent is **not constant** across a conversation. It's worst at turn 1 and improves with depth: SODA, a benchmark across conversation depths, shows agents become **9–52% safer** as task depth increases. Hidden-state probes confirm the activations migrate toward safety-aligned regions of representation space with interaction depth — an in-context alignment phenomenon. Mitigation is essentially free: **warm up the agent with a few benign tasks before exposing it to untrusted input**.

**Prereqs:** [_attacks.md](_attacks.md), [_jailbreaks.md](_jailbreaks.md)
**Related:** [cot-monitoring.md](cot-monitoring.md), [safety-case.md](safety-case.md), [../interpretability/README.md](../interpretability/README.md), [../agents/README.md](../agents/README.md)

---

## What it is

The empirical observation that an agent's safety behavior is depth-dependent. Concretely:

- At turn 1 (cold start), the agent's refusal / safe-handling rate on adversarial inputs is at its lowest.
- After 2–5 benign, on-task interactions (file inspection, normal tool calls, regular reasoning), the same adversarial prompt is refused or safely handled far more often.
- The improvement is large — 9–52 percentage points depending on attack and base model — and consistent across attack families.

This is a property of the *conversation state*, not the model weights. The same model is safer mid-session than it is at the start.

## How it works (mechanism)

The paper's interpretability probe shows that as the conversation grows, the hidden states migrate toward regions of activation space that are associated with safety-aligned behavior — regions the model already "knows" but doesn't enter from a cold prompt.

This is consistent with an in-context alignment story: SFT and RLHF teach the model what safety-aligned behavior looks like, but at the start of a session those representations are not active. Benign tool-calling tasks nudge the hidden state into the right region, after which adversarial inputs are interpreted from a safer prior.

### The SODA benchmark

SODA evaluates safety at *conversation depths* $d \in \{1, 2, 5, 10, \ldots\}$: an attack is preceded by $d-1$ benign agentic tasks (file operations, lookups, calculations) before the adversarial turn. Refusal / safe-handling rate is measured as a function of $d$.

### The warm-up mitigation

Before exposing a deployed agent to untrusted input, run it through a fixed sequence of benign agentic tasks (a "warm-up phase"). Empirically this closes most of the cold-start gap with no retraining, no extra weights, no architectural change.

## Why it matters

- **Almost every red-team study probes the agent at turn 1** — the worst possible time. Reported jailbreak rates likely *overstate* the production attack surface for warm-shoot agents.
- **Free deployment fix.** A warm-up phase costs a few extra tool calls per session and closes a measurable safety gap.
- **Clean interpretability target.** Hidden-state drift between cold and warm states is a sharp, measurable mechanism — a clean substrate for follow-on mechanistic work.
- **Reshapes how safety should be reported.** Safety numbers without specifying conversation depth are ambiguous; future benchmarks should report depth-conditional rates.

## Gotchas & tricks

- **The warm-up tasks must be genuinely benign and on-distribution.** Synthetic / pasted-in fake context doesn't replicate the effect — the hidden-state drift requires real reasoning over real tool outputs.
- **Effect size varies by attack family.** Out-of-distribution attacks (low-resource-language, character-encoding obfuscation) get smaller improvements; in-distribution paraphrased attacks get the largest.
- **Doesn't substitute for actual alignment training.** Closing the cold-start gap doesn't fix attacks the model would still fall for at depth 10 — it just removes the depth-1 amplification.
- **Production caveat.** If your agent restarts sessions frequently (e.g. per-turn statelessness for cost reasons), you re-pay the cold-start vulnerability each time.
- **Adversaries can target turn 1.** Now that the gap is known, attackers will preferentially probe at session start — defenders should reciprocally enforce a warm-up before serving untrusted input.

## Sources

- Paper: *The Cold-Start Safety Gap in LLM Agents* — Sun, Liu, Weng, UC San Diego, 2026 — [arXiv:2606.07867](https://arxiv.org/abs/2606.07867).
- Related: [_attacks.md](_attacks.md), [cot-monitoring.md](cot-monitoring.md).
