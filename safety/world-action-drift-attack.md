# World-Action Drift Attack
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An adversarial attack targeting **World-Action Models** (WAMs) — embodied models that jointly predict a future world state *and* emit an action. WAMs are marketed as safer than pure policies because the imagined future can be treated as a witness for the executed action (if the action was going to do something dangerous, the imagined future would show it). BadWAM (2026) shows this consistency assumption is fragile: small visual perturbations decouple the imagination from the action — the WAM continues to *dream right* while *acting wrong*.

**Prereqs:** [_attacks.md](./_attacks.md)
**Related:** [cot-monitoring.md](./cot-monitoring.md)

---

## What it is

A **World-Action Model** couples action generation with future world prediction. During inference the model produces two outputs from a shared representation: (a) the executed action $a_t$, and (b) an imagined next observation $\hat{o}_{t+1}$. The pairing is a safety story — if the model imagines a plausible future, and the executed action is what got the imagined future to make sense, then the action is self-witnessed.

A **World-Action Drift Attack** targets exactly this pairing. The attacker crafts a small, imperceptible perturbation of the input observation such that:

- The imagined future $\hat{o}_{t+1}$ remains plausible (the "dream" is unchanged).
- The executed action $a_t$ shifts to a target the attacker chose (the "act" is altered).

The imagination-based safety witness silently passes on an unsafe action.

## How it works

BadWAM (NUS / HK PolyU, 2026) formalizes and evaluates the attack:

1. **Threat model.** White-box (attacker has gradient access to the WAM). Attacker owns the *visual input channel* — a physical sticker, a doctored camera stream, a projector on a surface — but not the model, the environment, or the human overseer.
2. **Attack construction.** Standard PGD-style perturbation, but with a dual loss: (i) maximize the change in the action head's output toward a target action; (ii) minimize the change in the world head's predicted next observation. The joint minimization decouples the two heads' outputs.
3. **Attack transfer.** The perturbation transfers across seeds and across similar WAM checkpoints — not perfectly, but well enough to be a meaningful concern.
4. **Defense evaluation.** Adversarial training and input smoothing partially help but don't restore the imagination↔action guarantee — the shared representation *is* what the attack targets, and any defense at the input level pays a capability cost.

## Why it matters

- **Undermines a specific safety narrative around VLA / WAM systems.** The imagined future was being pitched as an internal witness for the executed action; BadWAM shows the witness is fragile against adversarial perturbations.
- **Re-motivates external verification.** For high-stakes embodied deployment, cross-checks that *don't* share the WAM's representation — an independent world simulator, a hard interlock on action magnitude, a separate outcome-safety classifier — are more robust than any intra-model consistency check.
- **Applies pressure on the "self-consistency as safety" pattern more broadly.** The same failure shape (a shared representation makes a self-witness attackable) appears in CoT monitoring and in other intra-model safety checks. The category is the risk, not just the WAM special case.

## Gotchas & tricks

- **Not the same as an adversarial-policy attack.** Classical adversarial attacks on RL policies flip the action; BadWAM flips the action *while preserving the imagined observation* — a strictly harder attack that also breaks the consistency check.
- **Defense at the shared representation is expensive.** Defending the world head or the action head alone doesn't help — the two heads share a body. Robustifying the shared representation is the correct place, but generic robust-training methods are expensive at foundation-model scale.
- **Transferable perturbations are more dangerous than white-box ones.** In deployment, an attacker rarely gets weight-level access. The transfer results are what make the attack a real-world concern.
- **Sim-to-real is the open question.** Reported results are simulator-heavy; the extent to which physical-world perturbations (stickers, projector patches) survive rendering pipelines is under-quantified.
- **This is a class, not an isolated exploit.** Any WAM/VLA that leans on internal-consistency-as-safety is a candidate — the paper's specific attack is one instance; expect a family.

## Sources

- Paper: *BadWAM: When World-Action Models Dream Right but Act Wrong* — Li, Yang, Wang, NUS / HK PolyU, 2026 — introduces the attack class and provides the BadWAM benchmark.
- Code / assets: [github.com/LiQiiiii/BadWAM](https://github.com/LiQiiiii/BadWAM).
- See also: [_attacks.md](./_attacks.md) for the broader attack taxonomy this fits into (adjacent to prompt-injection and agent-misuse for the embodied setting).
