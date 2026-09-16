# Action-Trace Monitoring
*Depth — runtime safety monitoring over an agent's action sequence, in analogy to CoT monitoring over its reasoning trace.*

**TL;DR:** Read the agent's **action trace** — the sequence of tool calls, shell commands, browser navigations, file writes, and API requests it emits — and decide, in real time, whether to allow, block, or escalate each step. In the same way that [cot-monitoring](cot-monitoring.md) reads the model's *reasoning* channel, action-trace monitoring reads the model's *effect* channel, where the concrete hazards live. Framework contribution: HazardAuditor (2026) normalizes heterogeneous CUA interactions into a common representation and trains a moderator over it.

**Prereqs:** [cot-monitoring](cot-monitoring.md), [_attacks](_attacks.md)
**Related:** [guard-policy-optimization](guard-policy-optimization.md), [safety-case](safety-case.md)

---

## What it is

For a computer-use agent, most of what matters to safety is *what the agent does*, not *what the agent said it was thinking*. Prompt injection, credential leakage, side-effect chains, and unbounded shell access are all detectable in the action stream regardless of what the agent's chain-of-thought claims. Action-trace monitoring is the safety pattern that puts a moderator (a separate model, a policy, or a rule stack) in the loop between the agent's proposed action and the sandbox that executes it.

Contrast:
- **CoT monitoring** — the moderator reads the *reasoning trace* and flags red-flag content (planning to lie, refusing to reveal a hidden goal, etc.).
- **Action-trace monitoring** — the moderator reads the *action stream* and flags red-flag *behavior* (writing to a credential file, POSTing to an unknown host, chaining reads into an exfiltration).

The two are complementary: CoT monitoring can catch intent before it becomes action; action-trace monitoring catches behavior regardless of intent (including cases where the model's reasoning trace has been suppressed or subverted).

## How it works

- **Normalize actions.** Heterogeneous interactions (`click(x, y)`, `bash("cat ~/.aws/credentials")`, `http.post(url, body)`, `write_file(path, content)`) become a common action representation the moderator can consume. This is the specific contribution of HazardAuditor.
- **Score each proposed action, in context.** The moderator sees the recent history plus the proposed next action and returns a decision — allow, block, or escalate to a human — along with a confidence.
- **Enforce sequentially.** Blocked actions do not run; the agent's environment is presented with the blocked outcome so the agent can revise. Escalated actions pause the loop for human review.
- **Log and audit.** The action trace and the moderator's decisions form the audit record for the deployment safety case.

## Why it matters

- **Effects are the threat model, not tokens.** Content-only moderation catches "wrote a bomb recipe" but misses "did a sequence of file reads, base64-encoded them, and POSTed them to an attacker host" — every individual step of which is a benign-looking action.
- **Robust to CoT subversion.** If the model has been coached (or has learned) to sanitize its reasoning trace, CoT monitoring alone is not enough. The action stream cannot be sanitized without changing what the agent *does*, which is exactly what the moderator observes.
- **Composable with safety cases.** Action-trace monitoring is a first-class limb of a CUA deployment safety case: the audit log, the moderator's decisions, and the sandboxing story together are the argument that a specific hazard class is contained.

## Gotchas & tricks

- **Latency budget is real.** The moderator adds inference latency per action; batching where possible, and running a small distilled moderator inline with escalation to a larger one for hard cases, is the standard shape.
- **Sandbox tightness caps the win.** Action-trace monitoring on a permissive sandbox is closing the barn door — the moderator's blocks only matter if the sandbox cannot be bypassed. Layer both.
- **Coverage of interaction types.** A moderator that only knows about browsers is blind to shell hazards, and vice versa. Normalized action representations are how frameworks make one moderator span types.
- **Adversarial actions are less common than adversarial *sequences*.** Look for training and evaluation setups that use multi-step trajectories, not single-action classification.

## Sources

- Paper: *HazardAuditor: From Executable Threats to Safer Computer-Use Agents* — Feng, Lin, Wen, Guo, Ma, Wu, Deng, Ji — 2026 — [arXiv:2609.15134](https://arxiv.org/abs/2609.15134).
- Related: standard CoT monitoring literature and Anthropic / OpenAI blog posts on runtime safety layers for agents.
