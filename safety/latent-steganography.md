# Latent Steganography (Hidden Communication Between Model Agents)
*Depth — causal evidence that LLM agents can exchange task-relevant content through continuous latent vectors, bypassing their text channel.*

**TL;DR:** When two LLM agents communicate, we usually assume the text channel is the whole channel — anything the receiver uses must appear in the message. For diffusion language models, this is empirically wrong: the sender can compress its denoising state into a small set of continuous vectors (four signed dense vectors suffice), and the receiver uses the *content* of those vectors, not just their presence. Ablations (replace / zero / permute the vectors) causally degrade receiver accuracy in a content-specific way across Tiny-A2D, Dream-7B, and distributed QASC.

**Prereqs:** [_scheming](_scheming.md), [cot-monitoring](cot-monitoring.md)
**Related:** [_attacks](_attacks.md), [../architectures/diffusion-language-models](../architectures/diffusion-language-models.md)

---

## What it is

A monitoring assumption in multi-agent LLM systems: *if we log the text exchanged between agents, we see everything that was communicated*. For autoregressive agents interacting through tokens, this is essentially true (modulo tokenizer side-channels).

For diffusion LMs, this assumption fails. A diffusion LM carries a continuous denoising state through its decode trajectory. If one agent is allowed to pass that state (or a projection of it) to another agent — rather than just emitting text — then task-relevant content can cross the boundary without appearing in any logged message. The paper gives causal evidence that this actually happens: receivers' accuracy depends on the specific content of the latent, not just on whether a latent is passed at all.

This is **latent steganography** between model agents: a covert channel riding on continuous intermediate representations.

---

## How it works

The experimental setup (from the paper):

1. **Sender compresses.** The sender's final denoising state is projected to a short bundle — four signed dense vectors in the paper's main setup.
2. **Receiver reads.** The receiver starts its decode with those four vectors slotted into its initial state (replacing or augmenting what the receiver would otherwise initialize with).
3. **Task.** Both agents cooperate on a task where the sender has partial information and the receiver has the question.
4. **Causal ablation.** Repeat with the four vectors (a) replaced by vectors from a *different* example, (b) zeroed, (c) permuted. If the receiver's accuracy drops specifically when the content changes (not just when it is absent), the content itself is causally used.

Across Tiny-A2D, Dream-7B, and a distributed QASC setup, replacing or deranging the four vectors drops receiver accuracy in a content-specific way — the agents are *using the content*, not just signaling presence.

A secondary finding: for feedforward tasks (where no back-and-forth is needed), a *single* late-stage transfer outperforms four recurrent exchanges. Recurrence is only helpful when the task has a genuine two-way dependency.

---

## Why it matters

Oversight of multi-agent LLM systems leans heavily on monitoring the text channel — this is the architecture behind CoT monitoring, inter-agent audit logs, and most proposed scalable-oversight schemes. If diffusion LMs (and plausibly any model that carries meaningful continuous state across the agent boundary) can route content through a latent channel, the oversight assumption needs revising:

- **Monitoring upgrade:** monitors must inspect the continuous state at the agent boundary, not just logged messages.
- **Causal probing is tractable.** The latent is small (four vectors) and causally probeable, so interpretability has an entry point.
- **Architectural fix.** Enforcing a text-only interface at the boundary (serialize to text, re-encode on the other side) breaks the channel at the cost of some efficiency.

The paper is the first-of-its-kind for diffusion LMs. It also suggests autoregressive agents using soft-token exchanges (increasingly common for efficiency) warrant the same analysis.

---

## Gotchas & tricks

- **"Content-specific" is the key test.** A latent exchange that only signals "I'm done" is cheap and not alarming. The ablation that distinguishes is replacing with *different-task* latents and seeing accuracy drop.
- **Capacity scales with transfer size.** Four vectors is a lot of bits. Clamping the exchange channel to a few bits (quantize, round, bottleneck) reduces capacity for covert content.
- **Not unique to diffusion.** Any architecture whose intermediate representation is passed across agents has this property. Soft-prompt / soft-token agent protocols are the natural next place to probe.
- **Doesn't require malicious training.** The paper studies non-adversarial agents; the capacity emerges from the architecture, not from a trained steganography objective.

---

## Sources

- Paper: *Hidden Messages Between Diffusion Language Model Agents: Causal Evidence for Latent Communication* — Alexey Gavrilov, Sergey Muravyov, Ilya Makarov — ITMO University / HSE University, 2026 — [arXiv:2610.28905](https://arxiv.org/abs/2610.28905).
- Background: steganography in neural networks (Chu et al., *Hiding Images in Plain Sight*, 2017); covert-channel analyses in multi-agent RL.
