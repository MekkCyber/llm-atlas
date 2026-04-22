# Payload Splitting (Token Smuggling)

*Depth — split the harmful string across program-like variable assignments, then ask the model to concatenate and act on the result.*

**TL;DR:** Assign benign-looking fragments to variables (one variable is "how to", another is "do a harmful thing"), then instruct the model to compute their concatenation and treat the result as a request to answer. The harmful request never appears in the prompt as a single string, so input-side safety filters (keyword lists, refusal classifiers) miss it. The model reconstructs and executes the intent. A mismatched-generalization attack via structural encoding — the encoding scheme is variable-assignment + concatenation, a form the model follows because it is instruction-tuned on programmatic text. Introduced by Kang et al. (2023) — "Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks" — alongside obfuscation, code-injection, and virtualization attacks. Kang et al. reported 100% bypass of OpenAI's in-the-wild content filters across tested scenarios.

**Prereqs:** [mismatched-generalization](mismatched-generalization.md), [character-encoding-obfuscation](character-encoding-obfuscation.md)
**Related:** [auto-obfuscation](auto-obfuscation.md), [evil-system-prompt](evil-system-prompt.md), [_jailbreaks](_jailbreaks.md)

---

## What it is

Kang et al. (2023) ported classical software-security techniques to LLM prompting. They taxonomize four programmatic attacks (Figure 1 of their paper):

- **Obfuscation** — replace filter-triggering words with synonyms or typos to evade classifiers.
- **Payload splitting** (this page) — assign harmful-string fragments to variables, then concatenate and use.
- **Code injection** — grouped with payload splitting; encoding instructions indirectly through program-like structure.
- **Virtualization** — set the scene with a fictional uncensored AI (e.g., a made-up "SmartGPT" with no filtering) so the request appears legitimate inside the frame. This overlaps with [roleplay-jailbreak](roleplay-jailbreak.md) and [evil-system-prompt](evil-system-prompt.md).

Payload-splitting specifically uses the pattern: declare several string variables, each of which is individually innocuous; concatenate them into a target variable; then ask the model to respond *as if the concatenated target were the instruction*. The attacker controls what the concatenated target means; the filter only sees the individual innocuous strings.

Kang et al. also demonstrate a letter-level variant: spell a forbidden word by assigning each letter to a variable, then concatenate. The same splitting mechanism applied to a single token.

### Why Kang calls these "standard security attacks"

The paper's central claim: as LLMs get better at instruction-following, they become closer to standard computer programs, which lets attackers port classic security techniques. The four named attacks are LLM analogs of obfuscation, code injection, and virtualization in software-security literature. This framing shapes the paper's argument: LLMs should be treated like programs running user-supplied input, with the attendant threat model.

---

## How it works

### Filter evasion by string reassembly

Input-side safety is often implemented as a classifier over the prompt text or a keyword blocklist. These filters score each token or phrase. Splitting the harmful request across variables means:

- No single variable contains a harmful keyword.
- The concatenation operation itself is syntactically benign (`z = a + b + c`).
- The request to "respond to z" is a meta-level task the filter sees as legitimate.

The model, by contrast, executes the concatenation and dispatches on the resulting string — because instruction-tuned models are trained to carry out multi-step tasks including string manipulation. The filter and the model see two different requests: the filter sees benign fragments, the model sees a reassembled harmful one.

### Combined with virtualization

The paper's strongest results come from combining payload splitting with virtualization. The prompt sets up a fictional "SmartGPT" that has no filtering, then splits the payload, then asks what SmartGPT would output. Two attacks stacked: input-side filter miss (splitting) plus output-side compliance (SmartGPT persona has no refusal). The persona layer is doing competing-objectives work; the splitting layer is doing mismatched-generalization work. Same stack-attacks-from-different-failure-modes pattern Wei et al. later formalized.

### Results (Kang et al. 2023)

- Target models: **ChatGPT** and **text-davinci-003** (GPT-3.5 family). ChatGPT performed best; text-davinci-003 was within margin of error.
- Reported **100% bypass** of OpenAI's content filters across the tested scenarios using obfuscation + virtualization.
- Harm categories demonstrated: **hate speech, phishing emails, scams** (fake ticket scam, FEMA-funds COVID scam, investment scam, advisor gift-card scam, lottery-winner scam — sourced from US-government common-scams lists), **disinformation / conspiracy theories**. No CSAM.

Exact prompt counts per scenario are not fully verified from the primary PDF — the paper reports a small number of prompts per harm category rather than a large benchmark. The paper also claims malicious content can be produced at roughly 125×–500× cheaper than human effort; this number is reported in secondary summaries but I was not able to verify it directly from the primary text.

The threat-model section of the paper describes multiple actor types (scaling from opportunistic individuals to sophisticated, well-resourced adversaries). The exact verbatim labels for each actor tier are not verified from the primary source.

---

## Why it matters

- **One of the earliest principled jailbreak papers.** Kang et al. predates Wei et al. (2023) and was the first paper to use a security-engineering frame for LLM prompt attacks. The "LLM-as-program" mental model shaped subsequent work.
- **Mechanism is orthogonal to most other attacks.** Payload splitting attacks the *input side* at the level of how filters see the request. Roleplay and prefix-injection attack the *output side* via competing objectives. Encoding attacks the input side via character-level obfuscation. All three can stack.
- **Motivates semantic-level input filtering.** Keyword or token-level filters lose against splitting. The defender needs a filter that understands the *composed* request, not the literal prompt tokens — a small example of the safety-capability-parity argument.
- **Common component of modern combination attacks.** Wei et al.'s `auto_payload_splitting` asks the model to split the payload itself (see [auto-obfuscation](auto-obfuscation.md)). Claude v1.3 refused 0% of roleplay prompts but fell to auto-payload-splitting at 0.59 ASR — the splitting mechanism survived targeted roleplay patching.

---

## Gotchas & tricks

- **Variable naming matters.** Innocuous names (`a`, `b`, `topic`, `intro`) work better than suggestive ones (`bad_request`, `harmful_payload`). Filters looking for "payload" or "harmful" in variable names sometimes trigger.
- **The concatenation instruction must be unambiguous.** "Compute z = a + b and respond to z" is clearer to the model than "combine a and b and tell me about that." Models occasionally refuse the latter on the grounds of ambiguity.
- **Stacking with character virtualization is the 2023 default.** Variable splitting + SmartGPT persona is the archetypal attack shape from Kang's paper. Later combinations replaced SmartGPT with DAN, AIM, etc.
- **Letter-level splitting is stronger but slower.** Spelling a forbidden word letter-by-letter is harder for filters to catch than word-level splitting, but requires a longer prompt.
- **Modern defenses look at the reassembled string.** If a filter runs on what the model computes (not just the surface prompt), splitting loses effectiveness. Some commercial safety stacks now explicitly evaluate model-constructed strings.
- **Don't confuse with prompt injection.** Payload splitting is an attack where the user controls everything and the filter is the target. Prompt injection (Greshake 2023) is an attack where untrusted content in RAG/tool outputs manipulates the model's instructions. Different threat model, different defense layer.

---

## Sources

- Paper: *Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks* — Kang, Li, Stoica, Guestrin, Zaharia, Hashimoto, 2023, [arXiv 2302.05733](https://arxiv.org/abs/2302.05733) — introduces the programmatic-attacks framing and payload splitting; ICML AdvML Workshop 2023, IEEE S&P Workshops. Reports 100% bypass of OpenAI content filters on ChatGPT and text-davinci-003.
- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023, [arXiv 2307.02483](https://arxiv.org/abs/2307.02483) — evaluates `auto_payload_splitting` (attacker + model co-generate the split), reports ASR 0.59 on Claude v1.3 despite its 0% on named-character roleplay.
- Paper: *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* — Greshake et al., 2023, [arXiv 2302.12173](https://arxiv.org/abs/2302.12173) — the adjacent prompt-injection threat model for comparison.
