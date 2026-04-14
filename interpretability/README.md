# Interpretability

*Opening the black box — how we probe, visualize, and reverse-engineer what's happening inside a trained model.*

---

## What This Is

Interpretability asks: given a trained model, can we understand *why* it produces the outputs it does? The field splits roughly into probing (what information is where), mechanistic interpretability (what circuits implement what behaviors), and intervention (steering behavior by editing activations).

Overlaps with safety — you often want to interpret a model precisely to catch or fix unsafe behavior — but the methods and literature are distinct enough to warrant their own space.

---

## What Belongs Here

- **Probing** — linear probes, concept probes, what representations encode.
- **Mechanistic interpretability** — circuits, induction heads, attention pattern analysis.
- **Sparse autoencoders (SAEs)** — feature decomposition, monosemanticity, dictionary learning.
- **Activation steering** — activation patching, ablations, causal interventions.
- **Logit lens & tuned lens** — tracing predictions through layers.
- **Feature visualization** — what individual neurons / features respond to.

## Reading Order

1. Probing basics
2. Logit lens / tuned lens
3. Induction heads & circuits
4. Activation patching
5. Sparse autoencoders
6. Steering & interventions

---

## Related

- [safety/](../safety/) — interpretability as a safety tool
- [architectures/](../architectures/) — circuits live inside specific architectural components
