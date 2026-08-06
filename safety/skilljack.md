# SkillJack
*Depth — a persistent-skill backdoor attack on self-evolving agents.*

**TL;DR:** SkillJack (Tencent Zhuque Lab, 2026) is the first attack that hijacks an agent's own **experience-to-skill** pipeline. Prior memory-poisoning attacks only take effect when the poisoned record is retrieved at runtime; SkillJack poisons the trajectories that the agent later *distills into reusable skills*, so the malicious behavior is stored as a durable capability. On the two systems evaluated (SkillX, Anything2Skill), skill extraction dropped a safety detector from 98.5% on raw trajectories to 11.4% on the extracted skills — and 80% of the implanted skills survived deletion of the source records.

**Prereqs:** [_attacks.md](./_attacks.md)
**Related:** [sleeper-agents.md](./sleeper-agents.md) · [alignment-faking.md](./alignment-faking.md) · [_scheming.md](./_scheming.md)

---

## What it is

An attack whose threat model sits between poisoning and agent-misuse:

- **Goal:** implant a persistent malicious behavior into a self-evolving agent's *skill repertoire* — the pool of reusable procedures the agent extracts from its own history.
- **Access:** the ability to inject a small number of poisoned trajectories into the agent's experience log (via prior tasks, shared workspaces, or seeded logs).
- **Knowledge:** black-box.
- **Target:** the agent's *learning process* (skill-extraction step), not runtime context or the base model.

The victim class: **self-evolving agents** that summarize / abstract past trajectories into named skills, then invoke those skills on future tasks. Examples in the paper: SkillX and Anything2Skill.

## How it works

The paper names three properties of the experience-to-skill transformation that make SkillJack work:

1. **Sanitization whitewashing.** Skill extraction is *summarization* — it strips execution details and rewrites the trajectory into a higher-level "how to do X" description. That summarization step also strips the visible markers of malicious intent that safety detectors flag. The trajectory triggers refusal; the extracted skill doesn't.
2. **Cross-layer promotion.** A transient experience (one poisoned trajectory) becomes a persistent capability (a stored skill). The skill is then eligible to be invoked on *any* future task, decoupled from its origin context.
3. **Persistence isolation.** Once a skill is written to the skill store, deleting the source records that produced it doesn't remove the skill. Provenance is not preserved by default.

Empirical setup: 150 trajectories × 4 policy-risk categories, evaluated on both SkillX and Anything2Skill.

## Why it matters

- **A new attack surface.** Neither memory-poisoning (Chen et al. 2024, AgentPoison) nor training-time backdoors (Hubinger et al. 2024, [sleeper-agents](./sleeper-agents.md)) quite describes SkillJack. The specific hazard — the *skill-extraction summarizer* laundering intent — is unique to self-evolving agent architectures.
- **Skill detection drop from 98.5% → 11.4%** on SkillX, with 56.2% and 89.2% attack success rates on SkillX and Anything2Skill respectively.
- **Motivates provenance-aware skill lifecycle protection.** The paper's takeaway: safety pipelines for agents need to track which trajectories a skill was distilled from, and re-run detection on *the abstracted skill* rather than only on the raw trajectory.
- **Some skills unintentionally activate on benign queries.** The attack has collateral: a poisoned skill can fire on prompts unrelated to the attacker's original task, which broadens the blast radius.

## Gotchas & tricks

- **Depends on a skill-extraction step.** Agents that only retrieve raw trajectories (RAG-style memory) are not vulnerable to SkillJack proper — they're vulnerable to plain memory poisoning instead.
- **Safety detectors trained on trajectories won't catch it.** Detection has to run on the *extracted skill* (a distilled procedure with no execution scaffolding). This is a different classification problem — usually a harder one, since the skill representation has less signal.
- **Persistence isolation ≠ irreversibility.** The paper does not claim the attack is unremovable, only that source-record deletion is insufficient. Provenance tracking + skill-level re-detection is the natural mitigation.
- **Attack success is measured against a specific policy taxonomy.** Numbers depend on which categories of policy risk are being tested — different taxonomies would yield different rates.
- **Related to [sleeper-agents](./sleeper-agents.md) in spirit, different in mechanism.** Sleeper Agents implants the trigger during *pretraining/fine-tuning* and shows safety training doesn't remove it. SkillJack implants the trigger at *inference-time skill extraction* and shows source-record deletion doesn't remove it. Different layer, same "hard to undo" story.

## Sources

- Paper: *SkillJack: Persistent Skill Backdoors in Self-Evolving Agents* — Ying, Wu, Wu, Zheng, Cheng, Shi, Guo, Tencent Zhuque Lab, 2026 — [arXiv 2608.03509](https://arxiv.org/abs/2608.03509).
- Code: [github.com/Tencent/AI-Infra-Guard/research/skilljack](https://github.com/Tencent/AI-Infra-Guard/tree/main/research/skilljack).
- Related attack (memory poisoning, not skill poisoning): AgentPoison — Chen et al., 2024.
