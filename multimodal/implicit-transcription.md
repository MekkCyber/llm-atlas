# Implicit Transcription in Speech LMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Interleaved speech–text language models — trained on sequences that mix speech and text tokens — go through an **implicit transcription** phase in their intermediate layers: the text-token spelling of the currently spoken word becomes decodable via the [logit lens](../interpretability/logit-lens.md), often as one of the top candidate tokens (up to 77% of utterances). This despite never being explicitly trained on speech-to-text. After transcribing, the model predicts the next word in the *text* space, then maps back to the speech domain.

**Prereqs:** [../interpretability/logit-lens.md](../interpretability/logit-lens.md), [README.md](./README.md)
**Related:** [../interpretability/README.md](../interpretability/README.md)

---

## What it is

Speech language models (SLMs) increasingly follow the "text-interleaving" recipe: pretrain on sequences that alternate speech tokens (from a speech codec/tokeniser) and text tokens, aiming to inherit text-LM capabilities in the speech modality. The internal dynamics of that mixing were unclear — do the two modalities share a common representation, or do they run in parallel through the network?

The implicit-transcription finding says: they *converge on text* mid-network. The speech input is silently rendered into its text-token form before the model does most of its downstream computation, then the text-space prediction is projected back to speech at the output.

## How it works

**The measurement.**
1. Take an interleaved speech–text LM.
2. Run a speech-only utterance through the forward pass.
3. At each layer, apply the model's *text* unembedding to the residual stream (a [logit lens](../interpretability/logit-lens.md) into the text vocabulary).
4. Read out top-k candidate *text* tokens at each layer.

**What the data show.**
- Intermediate layers put the text-token spelling of the current spoken word among the top candidates for up to **77%** of the tested utterances.
- The pattern holds across model families and sizes, though the peak depth varies (larger models transcribe deeper).
- Following the transcription peak, the residual stream shifts toward the *next-word* text token, then the final layers map back to speech tokens for the output.

**Drivers of the effect.**
- **Text-LM initialisation** — starting from a text-pretrained model dramatically increases how strongly implicit transcription emerges.
- **Interleaving data volume** — more interleaved data → stronger transcription phase.
- Correlates with the model's spoken-knowledge accuracy: models with stronger implicit transcription answer factual audio questions more accurately.

## Why it matters

- **Suggests a shared text backbone.** Interleaved SLMs appear to *use* text as their internal reasoning substrate — a strong argument that multimodal LMs currently converge on a text-shaped latent code.
- **Architecture design cue.** If text is the internal lingua franca, *deliberately* structured text-space intermediates (e.g. explicit transcription objectives) could be a training target for any-to-any models.
- **Interpretability wins from cross-modality lenses.** Applying a text unembedding to a speech residual stream is a template that transfers — image LMs and multimodal reasoning models may admit the same probe.

## Gotchas & tricks

- **Top-candidate rate, not always top-1.** The finding is that the transcription is *decodable*, not that it always wins the argmax. Downstream computation still uses distributional information.
- **Not all SLMs show it.** Models pretrained on speech-only or with weak text-interleaving can lack the phase — presence is a feature of the training recipe, not the architecture.
- **Correlation with accuracy is not causation.** Stronger transcription correlates with spoken-knowledge accuracy; whether *forcing* stronger transcription improves capability is a follow-up question.
- **The result is bidirectional in principle.** Same tooling applied to *text-only* inputs may reveal analogous implicit representations for other modalities in the reverse direction.

## Sources

- Paper: *Interleaved Speech Language Models Latently Work In Text* — Sternberg, Maimon, Adi (Hebrew University of Jerusalem), 2026 — [arXiv:2606.22473](https://arxiv.org/abs/2606.22473).
