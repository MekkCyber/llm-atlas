# Patch Reparameterization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pretrained semantic ViT's *final tokens* discard fine-grained visual detail — which blocks unified understanding + generation + editing from a single visual space. Patch Reparameterization keeps the original semantic pathway and **adds a reconstruction-aware patch embedding that feeds the same frozen ViT blocks**, so the same representation carries both semantic abstraction and pixel-recoverable detail. Introduced by **UniSpace** (2026); scaled to an 8B Mixture-of-Transformer-Experts model with no separate VAE pathway.

**Prereqs:** [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md)

---

## What it is

Unified multimodal models want one visual representation to (i) drive text-vision understanding, (ii) condition image generation, and (iii) support instruction-based image editing. The dominant obstacle: a semantic ViT (SigLIP, DINO, InternViT) trained for semantics discards pixel-level information by the final layers — the tokens work great for VQA but reconstruct blurrily.

UniSpace's diagnostic: the *frozen* transformer blocks of a semantic ViT are **not** intrinsically unable to preserve visual detail. The bottleneck is the input side — the **patch parameterization** — which drives the representation toward semantic abstraction. Patch Reparameterization fixes the input side while leaving the frozen blocks alone.

## How it works

Standard semantic ViT:

```
image → linear patch embed → [semantic] transformer blocks → final tokens
                                                              ↑
                                                    (pixel detail lost)
```

Patch Reparameterization adds a parallel embedding path that carries reconstruction information into the same frozen blocks:

```
image ┬→ semantic patch embed  ─┐
      │                          ├→ [frozen ViT blocks] → dual-purpose tokens
      └→ reconstruction patch ──┘
         embed (new, trainable)
```

The reconstruction patch embed is trained (semantic embed and ViT blocks stay frozen) so that the dual-purpose tokens support both:
- **Understanding heads** (unchanged) — semantic pathway still dominates.
- **A reconstruction head** — a lightweight decoder that regenerates the image from the tokens.

Both pathways share the same frozen ViT block stack — no separate VAE, no dual-tower architecture.

## Why it matters

- **One representation, three uses.** Understanding, generation, and editing all read/write the same token space. Simplifies unified-model training and eliminates the "generation VAE ↔ understanding ViT" impedance mismatch.
- **Frozen backbone.** The semantic ViT's understanding capability is preserved by construction, because its parameters are unchanged. Previous "unify by fine-tuning" approaches often traded understanding for generation quality.
- **8B validation.** UniSpace applies this to an 8B Mixture-of-Transformer-Experts and demonstrates practical text-to-image generation and instruction-based image editing in the same space.

## Gotchas & tricks

- **Reconstruction head design is not free.** The head must be strong enough to recover pixels from the dual-purpose tokens without themselves adding a VAE-scale parameter count — the whole point is avoiding a second heavy tokenizer.
- **Semantic vs reconstruction embed balance.** If the reconstruction embed dominates during training, semantic capability degrades even though the blocks are frozen (the tokens' distribution shifts). A careful loss balance is required.
- **Not a claim about any specific ViT.** The paper argues this works for semantic ViT blocks in general; specific behavior depends on which ViT is used as the frozen backbone.
- **Doesn't obviate tokenizers for other modalities.** Video and audio still need their own tokenization; Patch Reparameterization is a *visual* reparameterization.
- **Interaction with MoT-Experts routing.** Because the token distribution is dual-purpose, routing decisions in the downstream MoT-Experts stack must not implicitly separate "understanding tokens" from "generation tokens" — the whole point is that they're the same.

## Sources

- Paper: *UniSpace: Unified Visual Representation and Scalable Multimodal Modeling* — 2026 — introduces Patch Reparameterization and the 8B MoT-Experts model.
- Related: *Chameleon* (Meta, 2024), *Show-o* (2024), *Emu3* (2024), *Janus* (2024) — earlier unified-model designs with separate VAE pathways.
