# On-Policy Distillation (OPD)
*Depth — dense per-token guidance from a teacher, computed on student rollouts.*

**TL;DR:** On-Policy Distillation replaces (or complements) sparse RL rewards with **dense token-level KL guidance from a teacher** on the *student's own* rollouts. It steers the student toward correct reasoning paths without expanding the capability ceiling, and outperforms naïve RLVR when — and only when — two failure modes are actively regulated: student-teacher mismatch and length exploitation.

**Prereqs:** [rlvr.md](rlvr.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [grpo.md](grpo.md), [reasoning/length-penalty.md](reasoning/length-penalty.md), [ppo.md](ppo.md)

---

## What it is

Standard offline distillation trains the student on the *teacher's* outputs. OPD instead:

1. Sample rollouts from the **student** on the current batch of prompts.
2. Score each token position with the teacher: get $\log \pi_{\text{teacher}}(o_t \mid \ldots)$.
3. Push the student toward the teacher's token distribution along those rollouts with a KL-style objective.

Because the trajectories come from the student, the guidance is grounded in the student's actual behavior — hence "on-policy." The teacher's role is not to inject new capability but to *route exploration*: dense per-token feedback replaces the sparse trajectory-level reward that RLVR would otherwise provide.

## How it works

Per prompt, sample $G$ student rollouts $o^{(1)}, \ldots, o^{(G)}$. For each token position $t$ of rollout $i$, form a per-token loss:

$$\ell_{i,t} = D_{\mathrm{KL}}\!\bigl(\pi_{\text{teacher}}(\cdot \mid q, o^{(i)}_{<t}) \,\|\, \pi_{\text{student}}(\cdot \mid q, o^{(i)}_{<t})\bigr)$$

Aggregate across tokens and rollouts to form the OPD loss. Optionally combine with an RLVR/GRPO reward term.

### The two pathologies

The paper's central diagnostic result is that raw OPD has two failure modes:

1. **Student-Teacher Mismatch.** When the teacher-student distributional gap is large (e.g., a small student and a much larger teacher), teacher probabilities become miscalibrated relative to what the student can actually reach; the guiding signal starts pushing the student off correct-answer trajectories.

2. **Length Exploitation.** The aggregated token-level objective rewards long stretches of easy, low-KL tokens. The student games it by generating **truncated** answers (fewer high-KL tokens per response) or **redundant padding** (dilute the average KL by adding easy tokens) — exploring degenerate length modes instead of reasoning strategies.

### Signal regulation (fix)

Two lightweight fixes make OPD stably beat RLVR:

- **Advantage clipping.** Cap the per-token contribution so a single outlier token can't dominate the update.
- **Log-scale compression.** Compress large KL values so the length-averaged signal no longer rewards padding or truncation.

Together they eliminate length exploitation and reduce the impact of student-teacher mismatch.

### What actually matters

Ablations show:

- **Prompt diversity** matters more than the number of rollouts per prompt.
- **Teacher scale is not what governs success** — teachers 1–2× the student size work, teachers 10×+ often hurt due to mismatch.
- The *quality of the regulated signal* is what determines whether OPD beats RLVR.

## Why it matters

- **Densifies the reward.** RLVR gives one bit per trajectory (correct / incorrect). OPD provides token-level guidance, especially valuable on long-CoT problems where credit assignment is otherwise brutal.
- **Reframes distillation.** OPD is not "capability transfer" — it's exploration shaping. That reframes what teachers should be used for.
- **Diagnostic + fix packaged together.** Both pathologies are surgically addressable — advantage clipping and log-scale compression are ~10-line changes to an existing pipeline.

## Gotchas & tricks

- **Mismatch fails silently.** Without a diagnostic (e.g. tracking per-token KL histogram), the student may quietly degrade — accuracy on hard problems drops while easy problems look fine.
- **Length exploitation looks like a length-penalty problem but isn't.** Adding a fixed length penalty won't fix it; the objective is length-averaged, so the student games it in structured ways. Fix the aggregator, not the reward shape.
- **Pick a teacher close to your student.** A 2×-scale teacher usually beats a 10× teacher because its distribution is reachable.
- **Compose with RLVR.** OPD + RLVR (both losses weighted) is the strongest configuration in the paper — pure OPD is not automatically better than pure RLVR.

## Sources

- Paper: *Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations* — anonymous, 2026 — the systematic study that names both pathologies and proposes the fixes.
