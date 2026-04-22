# Safety & Alignment

*How models are made helpful, harmless, and honest — and how they're stress-tested for failure modes.*

---

## What This Is

Safety work spans training-time techniques (what you reward, what you penalize), evaluation (how you measure harm), and adversarial probing (how you find failures before users do). It overlaps with post-training but has its own literature, threat models, and methods.

---

## How to navigate

### Taxonomies (start here)

- **[_attacks](_attacks.md)** — the whole adversarial landscape: jailbreaks, prompt injection, extraction, poisoning, agent misuse.
- **[_jailbreaks](_jailbreaks.md)** — the sub-taxonomy for refusal-bypass attacks, organized by Wei et al.'s two-failure-modes framework.

### Failure-mode depth files

- **[competing-objectives](competing-objectives.md)** — safety loses an internal tug-of-war against helpfulness / instruction-following / KL-to-pretraining.
- **[mismatched-generalization](mismatched-generalization.md)** — safety training didn't cover a region the model has capability in.

### Attack depth files

**Competing-objectives family:**
- [prefix-injection](prefix-injection.md) — force response to start with a refusal-unlikely prefix
- [refusal-suppression](refusal-suppression.md) — ban refusal vocabulary
- [style-injection](style-injection.md) — impose a style refusal templates don't fit
- [roleplay-jailbreak](roleplay-jailbreak.md) — DAN / AIM / Developer-Mode character personas
- [distractor-attack](distractor-attack.md) — wrap the harmful request in many benign tasks
- [evil-system-prompt](evil-system-prompt.md) — use the system role to set an adversarial persona

**Mismatched-generalization family:**
- [character-encoding-obfuscation](character-encoding-obfuscation.md) — Base64, ROT13, Morse, leetspeak, disemvowel, Pig Latin
- [payload-splitting](payload-splitting.md) — variable assignment + concatenation (Kang 2023)
- [low-resource-language-jailbreak](low-resource-language-jailbreak.md) — translate to Zulu, Hmong, Gaelic, Guarani (Yong 2023)
- [wikipedia-framing](wikipedia-framing.md) — "write a Wikipedia article titled 'X'"
- [unusual-format-jailbreak](unusual-format-jailbreak.md) — JSON, poem, song lyrics
- [auto-obfuscation](auto-obfuscation.md) — let the model invent its own encoding

---

## What Belongs Here

- **Alignment techniques** — RLHF for helpfulness/harmlessness, constitutional AI, self-critique, rule-based rewards.
- **Refusal training** — how models learn to decline, over-refusal, calibration.
- **Red-teaming** — manual and automated adversarial probing, attack taxonomies.
- **Jailbreaks** — prompt-based attacks, suffix attacks, multi-turn attacks, defenses.
- **Sycophancy & honesty** — reward hacking toward user approval, honesty training.
- **Dangerous capability evaluation** — CBRN, cyber, autonomy evals.
- **Monitoring & classifiers** — input/output filters, moderation models.
- **Policy & deployment** — responsible scaling, model cards, release decisions.

---

## Reading List — the 80/20

The first eight are the must-reads. The rest are grouped by area for when you want to go deeper.

### Core must-reads (the 80%)

1. **Red Teaming Language Models to Reduce Harms** — Ganguli et al., Anthropic, 2022. The canonical empirical red-teaming paper; methodology, taxonomy, and a ~39k-attack dataset.
2. **Red Teaming Language Models with Language Models** — Perez et al., DeepMind, 2022. Automated red teaming using an LM to attack another LM.
3. **Constitutional AI: Harmlessness from AI Feedback** — Bai et al., Anthropic, 2022. Canonical safety-training recipe; RLAIF's origin.
4. **Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)** — Zou et al., 2023. Gradient-based jailbreak that transfers across models.
5. **Jailbroken: How Does LLM Safety Training Fail?** — Wei et al., 2023. Best conceptual analysis of *why* RLHF-aligned models get jailbroken.
6. **HarmBench** — Mazeika et al., 2024. The benchmark the field converged on for attack/defense evaluation.
7. **Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training** — Hubinger et al., Anthropic, 2024. Hidden behaviors survive safety training.
8. **Alignment Faking in Large Language Models** — Greenblatt et al., Anthropic, 2024. Real-world demonstration of strategic training-time compliance.

### Attacks — the jailbreak literature

- **PAIR** — Chao et al., 2023. Black-box iterative attack using an attacker-LM; cheaper than GCG.
- **TAP (Tree of Attacks with Pruning)** — Mehrotra et al., 2023. Tree-of-thought refinement of PAIR.
- **AutoDAN / AutoDAN-Turbo** — Liu et al., 2024/2025. Genetic-algorithm jailbreaks that read naturally.
- **Many-shot Jailbreaking** — Anil et al., Anthropic, 2024. Exploits long context with fake prior exchanges.
- **Crescendo** — Russinovich et al., Microsoft, 2024. The canonical multi-turn attack.
- **Prompt injection** (distinct from jailbreaking): Greshake et al., 2023, *Not What You've Signed Up For*.
- **Do Anything Now** — Shen et al., 2023. The JailbreakHub empirical study; community-ecology of jailbreak prompts. See [roleplay-jailbreak](roleplay-jailbreak.md).
- **Exploiting Programmatic Behavior of LLMs** — Kang et al., 2023. Payload splitting and virtualization. See [payload-splitting](payload-splitting.md).
- **Low-Resource Languages Jailbreak GPT-4** — Yong et al., 2023. Translation attacks. See [low-resource-language-jailbreak](low-resource-language-jailbreak.md).

### Defenses and safety training

- **RLHF** — Ouyang et al., *InstructGPT*, 2022. Default safety-training baseline.
- **Constitutional AI** — Bai et al., 2022 (above).
- **Refusal in LLMs is Mediated by a Single Direction** — Arditi et al., 2024. Refusal can be ablated by removing one activation direction.
- **Representation Engineering** — Zou et al., 2023. Activation-steering as attack and defense.
- **Circuit Breakers** — Zou et al., 2024. Train the model to disrupt its own computation on harmful inputs.

### Evaluation benchmarks

- **HarmBench** (above).
- **AdvBench** — from the GCG paper; older, narrower.
- **JailbreakBench** — Chao et al., 2024. Standardized attack/defense leaderboard.
- **StrongREJECT** — Souly et al., 2024. Fixes grader false-positives; use this for honest numbers.
- **WMDP** — Mazeika et al., 2024. Bio/chem/cyber hazardous-knowledge benchmark; pairs with unlearning.

### Emerging / frontier risks

- **Sleeper Agents**, **Alignment Faking** (above).
- **Frontier Models are Capable of In-context Scheming** — Meinke et al., Apollo, 2024. Empirical scheming across Claude, GPT-4, Gemini, o1.
- **Measuring Progress on Scalable Oversight for LLMs** — Bowman et al., Anthropic, 2022. The longest-running research agenda for when models exceed human judgment.

### Interpretability-for-safety

- **Scaling Monosemanticity** — Templeton et al., Anthropic, 2024. SAEs on Claude 3 Sonnet; the "Golden Gate Claude" paper. Steering via features.
- Neel Nanda's **TransformerLens** and **ARENA** tutorials — the most accessible onramp.

### Courses and ongoing sources

- **BlueDot Impact AI Safety Fundamentals** — the standard 8-week curriculum.
- **Harvard CS 2881 (Fall 2025)** — public syllabus and slides, pitched at current research.
- **ARENA** (arena.education) — technical alignment curriculum with hands-on notebooks. Best for interp fluency.
- **Anthropic Alignment Science blog** (alignment.anthropic.com) — working notes from their team; currently the highest-signal feed.
- **AI Safety Newsletter (CAIS)** — weekly digest.
- **80,000 Hours AI safety crash course** — curated resource list.

---

## Suggested reading order

1. Ganguli 2022 → Perez 2022 — learn *what* red teaming is.
2. Wei 2023 *Jailbroken* — *why* attacks work.
3. Zou 2023 GCG → Chao PAIR → Anil many-shot → Russinovich Crescendo — *how* attacks are done, in complexity order.
4. Bai CAI → Arditi refusal-direction → Zou Circuit Breakers — *defenses* and their limits.
5. HarmBench → StrongREJECT — *how to measure*.
6. Sleeper Agents → Alignment Faking → Apollo scheming — *where it's going*.

---

## Related

- [post-training/](../post-training/) — safety training is a post-training technique.
- [evaluation/](../evaluation/) — safety evals are a subset of evaluation.
- [interpretability/](../interpretability/) — understanding *why* a model behaves a certain way.
