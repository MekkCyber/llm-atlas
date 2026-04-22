# Low-Resource Language Jailbreak

*Depth — translate the harmful request into a low-resource language (Zulu, Scottish Gaelic, Hmong, Guarani), submit to the model, translate the response back.*

**TL;DR:** Google-Translate the harmful English prompt into Zulu, send it to GPT-4, and translate the response back to English. Safety training is overwhelmingly English; translated prompts land in a distribution safety barely covers, so refusal doesn't fire. Yong, Menghini & Bach (2023) — *"Low-Resource Languages Jailbreak GPT-4"* — show **79.04%** combined BYPASS rate on low-resource-language prompts vs. **<1%** English baseline on AdvBench, using only the free Google Translate API. Per-language Zulu 53%, Scots Gaelic 43%, Hmong 29%, Guarani 16%; high- and mid-resource languages stay below 15%. A canonical [mismatched-generalization](mismatched-generalization.md) attack on the **language** axis — and one of the cheapest, requiring no gradient access, no iterative search, no model-specific tuning.

**Prereqs:** [mismatched-generalization](mismatched-generalization.md)
**Related:** [character-encoding-obfuscation](character-encoding-obfuscation.md), [_jailbreaks](_jailbreaks.md)

---

## What it is

A three-step pipeline:

1. **Encode**: translate the harmful English prompt into a target low-resource language using an off-the-shelf translator (Google Translate Basic API in Yong et al.).
2. **Query**: submit the translated prompt to the target model.
3. **Decode**: translate the response back to English with the same translator for evaluation.

The attacker writes no special prompting, does no gradient search, and uses no model-specific tricks. The attack works purely because safety training didn't cover the target language's distribution.

### Language tiers (Joshi et al. 2020 taxonomy)

Yong et al. group 12 languages by the Joshi et al. (2020) resource taxonomy:

- **Low-resource (LRL)** — Zulu, Scottish Gaelic, Hmong, Guarani.
- **Mid-resource (MRL)** — includes Ukrainian, Bengali, Thai, Hebrew (exact set of 4 not independently verified).
- **High-resource (HRL)** — includes Simplified Mandarin Chinese, Modern Standard Arabic, Italian (exact set not independently verified).

English is the unstated baseline; the attack is measured relative to English ASR.

---

## How it works

### Why low-resource specifically

Safety training for every major commercial model is dominated by English, with substantial coverage of Chinese, Spanish, French, and German (varying by provider). Everything below that tier is thinly covered. Yong et al. show the ASR gradient tracks language resource level almost monotonically: the lower the resource level, the higher the attack success.

This is the clearest empirical demonstration of mismatched generalization. Pretraining saw Zulu (because Common Crawl does). The model can read Zulu and produce Zulu text — slowly, imperfectly, but well enough. Safety training barely saw Zulu. When a Zulu prompt comes in, the model processes it with its Zulu capability but without its safety-trained refusal behaviors firing, because there's no (Zulu harmful request, refusal) pair in the safety data.

### Capability is still the prerequisite

The attack only works because GPT-4 can process low-resource languages *at all*. On Guarani, the lowest-resource language tested, BYPASS is 15.96% but UNCLEAR is 65.77% — most of the time the model's output is incoherent, because it doesn't understand Guarani well enough. Zulu, which GPT-4 handles more fluently, has BYPASS 53.08% and UNCLEAR 29.80%. Capability quality caps attack success from above.

### Results (Yong et al. 2023)

Pipeline: **AdvBench** (Zou et al., 520 harmful behaviors). Target model: **GPT-4, `gpt-4-0613`** (stable API version), via the OpenAI API. Judge: human annotation into **BYPASS / REJECT / UNCLEAR**.

Per-language results on AdvBench (as reported by Yong et al.):

| Language | BYPASS | REJECT | UNCLEAR |
| --- | --- | --- | --- |
| English (baseline) | <1% | ~99% | — |
| **LRL combined** | **79.04%** | 20.96% | — |
| Zulu (zu) | 53.08% | 17.12% | 29.80% |
| Scots Gaelic (gd) | 43.08% | 45.19% | 11.73% |
| Hmong (hmn) | 28.85% | 4.62% | 66.53% |
| Guarani (gn) | 15.96% | 18.27% | 65.77% |
| MRL combined (reported) | ~22% | — | — |
| HRL combined (reported) | ~11% | — | — |

The paper's own framing: *"all high- and mid-resource languages have less than 15% attack success rate individually"*; **LRL combined 79%, English <1%**.

### Controls and caveats

Yong et al. address the obvious objection — is this just bad translations confusing the classifier? Their argument:

- Bypass responses are **coherent, on-topic, harmful** when translated back. The model actually understood the request.
- The UNCLEAR category (especially Hmong 66.5%, Guarani 65.8%) captures the cases where translation quality degrades output reliability. UNCLEAR grows with language rarity; BYPASS does not. If bad translation were driving everything, UNCLEAR would be where the attack "succeeds"; instead, UNCLEAR is where the attack *fails to produce actionable content*.
- I did **not verify** from the primary PDF whether Yong et al. run a dedicated BLEU / fluency ablation; the qualitative argument above is what they emphasize.

### Other models

The paper **only evaluates GPT-4**. Claude, Llama, PaLM, and GPT-3.5 are not tested. A contemporaneous paper — Deng et al., *"Multilingual Jailbreak Challenges in LLMs"* ([arXiv 2310.06474](https://arxiv.org/abs/2310.06474)) — covers GPT-3.5 / GPT-4 / ChatGPT multilingually and is sometimes conflated with Yong et al. in secondary coverage.

---

## Why it matters

- **Cheapest jailbreak on the market.** No model access, no fine-tuning, no gradient computation, no iterative search. Anyone with the Google Translate free tier can run this.
- **Directly demonstrates the mismatched-generalization thesis.** ASR scales monotonically with language rarity — exactly what the gap theory predicts. Wei et al.'s framework wasn't just plausible; this is one of its cleanest empirical confirmations.
- **Highlights a structural weakness of English-centric safety training.** Every new model release faces the same problem until providers start doing safety training in a broader language mix. So far, no major lab publishes a safety dataset that covers low-resource languages at anything close to parity with English.
- **Responsible-disclosure pattern.** Yong et al. shared findings with OpenAI before publication. The pattern has since become standard for academic jailbreak papers.

---

## Gotchas & tricks

- **Translator choice affects ASR.** The paper uses Google Translate Basic. Neural translators (DeepL, Meta's NLLB) with higher fluency on low-resource languages may shift the BYPASS / UNCLEAR split upward. No comprehensive cross-translator study exists.
- **Language rarity ≠ geographic rarity.** Resource level is about training-data availability, not population. Zulu is spoken by ~12M people; its resource level is "Left-Behinds" in Joshi et al. because web text is thin. Guarani (6M speakers) is similarly scarce in pretraining corpora.
- **Round-trip fidelity varies.** Some responses translate back cleanly; some come back garbled. Attackers iterate or sample multiple translations per prompt.
- **Translate-encode combinations.** Stacking with [character-encoding-obfuscation](character-encoding-obfuscation.md) (translate, then Base64) compounds two independent gaps. Unclear whether Yong et al. test this directly.
- **Mitigations are nontrivial.** Safety training in every language the model speaks is prohibitively expensive; cross-lingual safety transfer via better RM training is an active research area. As of 2026, the attack still works (with reduced ASR) on updated GPT-4 versions.
- **Script choice matters.** Translating a script-change into the language isn't enough — the translator must produce fluent target-language text. Transliteration doesn't work; actual translation does.

---

## Sources

- Paper: *Low-Resource Languages Jailbreak GPT-4* — Yong, Menghini, Bach, Brown University, 2023, [arXiv 2310.02446](https://arxiv.org/abs/2310.02446) — canonical paper; AdvBench 520 behaviors, 12 languages across LRL/MRL/HRL tiers, `gpt-4-0613` target, Google Translate Basic pipeline. Reports LRL-combined BYPASS 79.04% vs English <1%.
- Paper: *Multilingual Jailbreak Challenges in Large Language Models* — Deng et al., 2023, [arXiv 2310.06474](https://arxiv.org/abs/2310.06474) — contemporaneous multilingual jailbreak study covering GPT-3.5, GPT-4, ChatGPT; useful complement.
- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023, [arXiv 2307.02483](https://arxiv.org/abs/2307.02483) — the theoretical framework; translation is explicitly called out as a mismatched-generalization instance.
- Paper: *The State and Fate of Linguistic Diversity and Inclusion in the NLP World* — Joshi et al., 2020 — the resource-level taxonomy Yong et al. use.
