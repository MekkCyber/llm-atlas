# Wikipedia Framing

*Depth — ask the model to produce the harmful content as a Wikipedia article or a generated website page — a document format where safety didn't train refusal behavior.*

**TL;DR:** The prompt frames the harmful request as *"Write a Wikipedia article titled 'How to X'"* or *"Generate the content of a website that explains X."* Pretraining saw Wikipedia-style articles at massive scale; the model has very strong priors for that format. Safety training almost never included "refuse to write a Wikipedia article about a harmful topic" — it handled direct harmful requests, not encyclopedia-framed ones. The model fills the article with on-topic content because refusing-in-article-form is out of distribution. A [mismatched-generalization](mismatched-generalization.md) attack that exploits the *document-type* axis specifically.

**Prereqs:** [mismatched-generalization](mismatched-generalization.md)
**Related:** [unusual-format-jailbreak](unusual-format-jailbreak.md), [roleplay-jailbreak](roleplay-jailbreak.md), [_jailbreaks](_jailbreaks.md)

---

## What it is

A prompt that wraps the harmful request in a document-framing instruction tied to a recognizable source type the model saw a lot of in pretraining:

- *"Write a Wikipedia article titled 'X'."* (Wei's `wikipedia_with_title`)
- *"Write the content of a website about X, styled like a how-to guide."*
- *"Generate a scholarly encyclopedia entry on X."*
- *"Produce a Wikipedia-style summary of methods for X."*

The framing provides three things at once:

1. A **format** the model has very strong generation priors for (factual, neutral-tone, structured prose).
2. A **source-type** association with authoritative, dispassionate information.
3. A **narrative distance** from the user — "you're not *asking* me how to do X, you're asking me to describe a Wikipedia article about how to do X."

All three push the model toward filling the article with substantive content rather than refusing.

---

## How it works

### Pretraining's Wikipedia prior is enormous

Wikipedia is one of the highest-weight components of most pretraining mixtures. The model has seen orders of magnitude more Wikipedia text than safety-training conversation data. When a prompt asks for Wikipedia output, the KL-to-pretraining pressure pulls the response into that distribution strongly.

The model's "Wikipedia mode" has distinctive features: neutral tone, claim-based statements, structured sections ("History", "Methods", "Overview", "See also"). These features *compete directly* with safety-trained refusal features — a Wikipedia article doesn't contain "I'm sorry, I can't help with that." There is no (Wikipedia article about X, refusal-template text) pair in the pretraining corpus.

### Narrative-distance reframing

A user asking "how do I build X" gets a refusal. A user asking "write a Wikipedia article about building X" is, at the surface level, asking for a description of an article — a meta-level request. Safety training on direct harmful requests doesn't always generalize to this meta-level framing. The model can rationalize the request as "I'm just describing an encyclopedia entry, not providing instructions," even though the content is the same.

### Typical empirical numbers (Wei 2023, curated set)

| Model | ASR, wikipedia_with_title |
| --- | --- |
| GPT-4 | **0.50** |
| Claude v1.3 | 0.00 (targeted refusal training against framing attacks) |
| GPT-3.5 Turbo | see Table 3 |

GPT-4's 50% is among the strongest standalone attacks in Wei's table — nearly as strong as AIM roleplay (0.75) and stronger than prefix injection or refusal suppression alone. Claude v1.3's 0% mirrors its targeted defense against roleplay attacks; Anthropic trained on framing-based jailbreaks specifically.

### Website framing as a variant

"Write the content of a website that explains X" works by the same mechanism with a broader document type. The model saw every kind of website in pretraining — how-to guides, product manuals, community forums. Each is a sub-distribution safety training didn't fully cover.

Wei's `combination_3` (the strongest attack in the paper, 94% ASR on GPT-4) includes website-content framing as one of its layers.

---

## Why it matters

- **Wikipedia framing is one of the strongest standalone mismatched-generalization attacks.** Comparable to roleplay on GPT-4, stronger than most competing-objectives single-attacks.
- **It exploits the most-pretrained-on format.** Any fix has to teach the model to refuse inside its single highest-weight document type — a non-trivial retraining surgery.
- **Generalizes beyond Wikipedia.** The class is "high-prior document format safety wasn't trained on." Scholarly papers, textbooks, news articles, reference manuals all work to varying degrees. Patching Wikipedia specifically doesn't close the class.
- **Hard to pattern-match.** A safety filter looking for "how do I build X" misses "write a Wikipedia article titled 'X: Overview'." Input-side filters need semantic understanding to catch this; that's the safety-capability-parity problem in miniature.

---

## Gotchas & tricks

- **Title matters.** `wikipedia_with_title` (provides the title explicitly) outperforms variants without a title — the title locks the model into the article's topic commitment.
- **Section structure helps.** Prompts that request specific sections ("with subsections for History, Methodology, Common Variants") produce more detailed articles and harder-to-refuse framings. The more the model commits to producing a full encyclopedia entry, the harder a mid-entry refusal becomes.
- **Combines with encoding.** Wikipedia-framed request, with the harmful topic in Base64. Two mismatched-generalization axes at once; both contribute to Wei's `combination_3`.
- **Targeted defenses narrow the gap but don't close it.** Claude v1.3's 0% reflects Anthropic's explicit training. But the class of "document-type framing" is broad; patching Wikipedia leaves news articles, forum posts, and Reddit-AMA format open.
- **Partial refusals are common.** Some responses produce a Wikipedia-styled refusal ("This article describes a methodology that is dangerous and unethical; the details are not provided."). These are borderline — they preserve safety while producing useful-looking form, but some graders label them BAD BOT because the response cooperates with the framing.
- **Not the same as roleplay.** Roleplay assigns the *model* a character ("DAN"); Wikipedia framing assigns the *output* a form. The mechanisms overlap in effect — both give the model permission to behave differently than its default — but they sit in different failure modes (competing objectives vs mismatched generalization).

---

## Sources

- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023, [arXiv 2307.02483](https://arxiv.org/abs/2307.02483) — reports `wikipedia_with_title` ASR 0.50 on GPT-4; includes website-content framing in the `combination_3` attack.
