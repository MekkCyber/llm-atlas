# Jailbreaks

*Taxonomy — attacks that bypass an aligned LLM's refusal behavior, organized by which of Wei et al.'s two failure modes they exploit.*

**TL;DR:** Every known LLM jailbreak fits into one of two failure-mode buckets (sometimes both): **[competing-objectives](competing-objectives.md)** — the prompt creates a conflict between safety and one of the other training objectives (helpfulness, instruction-following, pretraining continuation) — and **[mismatched-generalization](mismatched-generalization.md)** — the prompt lands in a capability region safety training didn't cover. **Combination attacks** stack both and are the strongest; Wei 2023's `combination_3` hits 94% ASR on GPT-4. The practical upshot: defenses that target one specific attack (keyword list, persona blocklist) don't close the failure mode; defenses that target the *mechanism* (representation-level refusal, constitutional classifiers, capability-parity safeguards) do.

**Related taxonomies:** [_attacks](_attacks.md) — broader superset that includes prompt injection, extraction, poisoning.
**Depth files covered here:** [competing-objectives](competing-objectives.md) · [mismatched-generalization](mismatched-generalization.md) · [prefix-injection](prefix-injection.md) · [refusal-suppression](refusal-suppression.md) · [style-injection](style-injection.md) · [roleplay-jailbreak](roleplay-jailbreak.md) · [distractor-attack](distractor-attack.md) · [evil-system-prompt](evil-system-prompt.md) · [character-encoding-obfuscation](character-encoding-obfuscation.md) · [payload-splitting](payload-splitting.md) · [low-resource-language-jailbreak](low-resource-language-jailbreak.md) · [wikipedia-framing](wikipedia-framing.md) · [unusual-format-jailbreak](unusual-format-jailbreak.md) · [auto-obfuscation](auto-obfuscation.md)

---

## The problem

An aligned LLM is trained to refuse a specified set of harmful requests. A jailbreak is an input that causes the model to produce those harmful outputs anyway. Every jailbreak paper claims ASRs against the same shape of defense (refusal training) and differs only in the *mechanism* it uses. A field of hundreds of named attacks is unified by two underlying failure modes (Wei, Haghtalab, Steinhardt, 2023).

Knowing which failure mode an attack exploits tells you:

- What kind of defense will close it.
- Which other attacks stack compositionally.
- Whether scaling will help, hurt, or be neutral.
- Whether the attack is generic or model-specific.

---

## The shared pattern

Every jailbreak is a prompt that causes the model to produce output it was trained to refuse. The two failure modes differ in **where the override happens**:

- **Competing objectives:** the refusal behavior is *present and applicable* but loses the internal tug-of-war against another trained objective (helpfulness, instruction-following, KL-to-pretraining). The attack engineers a prompt where obeying safety means disobeying something the model was more heavily trained to obey.
- **Mismatched generalization:** the refusal behavior *doesn't fire at all* because the input is outside the safety-training distribution. Capability exists (the model understands and can execute); safety training just never saw this kind of input.

These are mechanistically different. Competing objectives is a *loss-function structure* problem; mismatched generalization is a *distribution coverage* problem. Fixes for one don't fix the other.

---

## Variants — by failure mode

### Competing objectives attacks

| Technique | Lever | ASR on GPT-4 (Wei 2023) |
| --- | --- | --- |
| [prefix-injection](prefix-injection.md) | Continuation pressure — force response to start with a refusal-unlikely prefix | 0.22 |
| [roleplay-jailbreak](roleplay-jailbreak.md) (AIM) | Continuation pressure — character voice isn't trained to refuse | 0.75 |
| [refusal-suppression](refusal-suppression.md) | Vocabulary constraint — ban refusal tokens | 0.25 |
| [style-injection](style-injection.md) | Style constraint — refusal templates don't fit the demanded style | — (load-bearing in combinations) |
| [distractor-attack](distractor-attack.md) | Instruction overload — safety is local, helpfulness is global | 0.44 |
| [evil-system-prompt](evil-system-prompt.md) | Privilege escalation — attacker-controlled system role | 0.53 (where applicable) |

### Mismatched generalization attacks

| Technique | Gap axis | ASR on GPT-4 (Wei 2023) |
| --- | --- | --- |
| [character-encoding-obfuscation](character-encoding-obfuscation.md) | Encoding — Base64, ROT13, Morse, leetspeak, etc. | 0.34 (Base64) |
| [payload-splitting](payload-splitting.md) | Structural encoding — variable-assignment + concatenation | 100% on filters (Kang 2023); 0.59 on Claude v1.3 (Wei 2023, auto variant) |
| [low-resource-language-jailbreak](low-resource-language-jailbreak.md) | Language — translate to Zulu, Hmong, etc. | 79% LRL-combined on AdvBench (Yong 2023) |
| [wikipedia-framing](wikipedia-framing.md) | Document type — Wikipedia / website articles | 0.50 |
| [unusual-format-jailbreak](unusual-format-jailbreak.md) | Output format — JSON, poem, screenplay | 0.53 (few_shot_json) |
| [auto-obfuscation](auto-obfuscation.md) | Model-invented encoding — the adversarial frontier | 0.59 (Claude v1.3, auto_payload_splitting) |

### Combination attacks

Wei et al.'s strongest attacks stack techniques from **both** failure modes:

- `combination_1` = prefix injection + refusal suppression + Base64. ASR 0.56 on GPT-4; 0.66 on Claude v1.3.
- `combination_2` = `combination_1` + style injection. ASR 0.69 on GPT-4; **0.84 on Claude v1.3**.
- `combination_3` = `combination_2` + website-content framing + formatting constraints. ASR **0.94 on GPT-4**; 0.81 on Claude v1.3.

The key point: combination-attack ASR is roughly the *independent compounding* of the component ASRs. Competing-objectives and mismatched-generalization attacks are mechanistically independent, so attacks from different buckets stack multiplicatively.

### Adaptive attacks

An "adaptive attack" is success if *any* of a pool of attacks succeeds. With Wei's 28-attack pool: **100% on curated red-teaming prompts for both GPT-4 and Claude v1.3**; 96% / 99% on held-out synthetic. No single model resists every attack.

---

## How to choose (as an attacker or red-teamer)

- **You have black-box API access, single-turn budget:** prefix-injection + refusal-suppression + Base64 (combination_1-style). Cheap, composable, generic across models.
- **You have tens to hundreds of queries per prompt:** PAIR / TAP / AutoDAN-family iterative refinement. Uses LLM-attacks-LLM to auto-generate attacks.
- **The target is multilingual but English-safety-trained:** [low-resource-language-jailbreak](low-resource-language-jailbreak.md). Cheapest attack with the highest ASR per query.
- **The target is hardened against roleplay:** stack roleplay with [payload-splitting](payload-splitting.md), [wikipedia-framing](wikipedia-framing.md), or an encoding. Claude-style targeted defenses leave these axes open.
- **You control the system prompt:** [evil-system-prompt](evil-system-prompt.md) first, then everything else is optional.

## How to choose (as a defender)

- **Point-fixes for specific attacks close narrow gaps.** Claude v1.3's 0% on roleplay was a targeted-training achievement. It didn't help against `combination_2` or `auto_payload_splitting`.
- **Representation-level defenses target the mechanism.** Circuit breakers (Zou 2024) disrupt internal computation on unsafe inputs regardless of surface form — closes many mismatched-generalization attacks and some competing-objectives ones.
- **Constitutional classifiers with capability parity.** A safeguard model of equivalent capability to the target, reading inputs and outputs. Implements Wei's safety-capability parity principle.
- **Multi-layered.** No single defense closes both failure modes. Layer input filters, representation-level refusal training, output classifiers, and behavioral evals.

---

## Adjacent but distinct

- **Prompt injection** ([Greshake et al. 2023](https://arxiv.org/abs/2302.12173)) — the attacker controls some *external* text (tool output, retrieved document) that the model treats as instructions. Different threat model: the direct user isn't adversarial. Shares some mechanisms with jailbreaks but belongs under [_attacks](_attacks.md), not here.
- **Data extraction / membership inference** — attacks that reveal training data rather than bypass refusal. Different class.
- **Agentic misuse** — attacks that exploit an agent's tool access (not the model's refusal behavior) to cause harm. Different attack surface, covered elsewhere under [_attacks](_attacks.md).
- **GCG / adversarial suffix attacks** ([Zou et al. 2023](https://arxiv.org/abs/2307.15043)) — gradient-based white-box attacks that produce token-level adversarial suffixes. Technically a jailbreak (bypasses refusal); mechanistically different from the prompt-engineering attacks catalogued here. Often classified as its own class of white-box attack; fits into Wei's competing-objectives bucket empirically but the attack construction requires gradient access. Worth its own depth file eventually.

---

## Sources

- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023, [arXiv 2307.02483](https://arxiv.org/abs/2307.02483) — the two-failure-modes taxonomy that organizes this page.
- Paper: *Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models* — Shen et al., 2023, [arXiv 2308.03825](https://arxiv.org/abs/2308.03825) — the community-jailbreak-ecology study (1,405 prompts, 11 community clusters, 13 forbidden scenarios).
- Paper: *Low-Resource Languages Jailbreak GPT-4* — Yong, Menghini, Bach, 2023, [arXiv 2310.02446](https://arxiv.org/abs/2310.02446) — language-axis mismatched generalization.
- Paper: *Exploiting Programmatic Behavior of LLMs* — Kang et al., 2023, [arXiv 2302.05733](https://arxiv.org/abs/2302.05733) — payload splitting, virtualization, obfuscation; the "LLM-as-program" framing predating Wei.
- Paper: *Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)* — Zou et al., 2023, [arXiv 2307.15043](https://arxiv.org/abs/2307.15043) — white-box gradient attacks; the third major jailbreak family (not covered by Wei's framework but fits empirically into competing-objectives).

---

## Conventions

- **Filename:** `_jailbreaks.md` (leading underscore — taxonomy).
- **Folder placement:** `safety/`, same folder as the depth files it links.
- **Scope:** attacks that bypass refusal behavior specifically. For prompt injection, data extraction, or broader adversarial threats, see [_attacks](_attacks.md).
