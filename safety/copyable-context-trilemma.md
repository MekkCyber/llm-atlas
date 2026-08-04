# Copyable-Context Safety Trilemma
*Depth — a formal impossibility result for context-based safeguards on dual-use tasks.*

**TL;DR:** If an LLM safeguard decides whether to answer using only *copyable* evidence — message text, tool history, anything an attacker can imitate — then for dual-use tasks it cannot simultaneously provide **useful capability**, **reliable safety**, and **open access**. The paper derives the exact worst-case attacker-assist floor under this evidence model and shows that **trusted credentials** (hard-to-copy attestations of downstream use) are the only path to escape it — under an additional identifiability condition.

**Prereqs:** [_jailbreaks.md](./_jailbreaks.md)
**Related:** [safety-case.md](./safety-case.md), [mismatched-generalization.md](./mismatched-generalization.md)

---

## What it is

A theoretical framing that separates two things content-safety practice conflates: the **capability** the model releases into a conversation, and the **evidence** available to decide whether to release it. When the evidence is copyable, an attacker can construct any authorized-user pattern that a benign professional would produce, so the safeguard cannot distinguish between them. The consequence is a **trilemma**: useful capability + reliable safety + open access are mutually incompatible.

## How it works

**Setup.** A dual-use task has an authorized-user distribution `D_auth` and an attacker distribution `D_att` over conversations. A safeguard `S: context → {allow, deny}` decides based on the context.

**Key claim.** If the context is copyable — attackers can imitate any distinguishing feature — then `D_att ⊇ D_auth` in the safeguard's decision space. Any allow-rule that helps authorized users also helps attackers proportionally.

**Trilemma.** The paper derives the exact worst-case attacker-assist floor under this model. Given desired usefulness `U` and openness `O`, safety `Safe` is upper-bounded — it cannot be pushed to 1 without giving up either usefulness or openness.

**Escape hatch.** Add a **trusted credential** — a hard-to-copy attestation carried by an authorized user that correlates with genuine downstream use. This introduces non-copyable evidence. Under an additional condition (credential identifiability), the floor can be eliminated.

## Why it matters

- Sharpens the open-weight and open-access safety debate: pure prompt-based filtering is provably insufficient for dual-use tasks — no more classifier can close the gap.
- Redirects safeguard research: from *detecting the bad request* to *attesting the use*. That requires infrastructure (credentials, verification, deployment programs), not just better ML.
- Provides a formal footing for the observed empirical result that adaptive attackers eventually break every content-only safeguard.

## Gotchas & tricks

- The trilemma applies to *dual-use* tasks specifically — tasks where the answer is either good or bad depending only on the actor. Purely-benign tasks and purely-malicious tasks are unaffected.
- The trusted-credential escape requires the credential to be predictive of genuine downstream use *and* hard to spoof — neither is trivial. Real trusted-access programs (developer verification, research-access programs) partially satisfy both.
- The result is worst-case; typical attackers may not reach the floor. But the floor exists and is the right thing to measure safeguards against.

## Sources

- Paper: *Safeguards Based on Copyable Context Cannot Provide Reliable Safety for LLMs* — 2026 — [arXiv:2607.27951](https://arxiv.org/abs/2607.27951)
