# Reward Modeling (for LLM post-training)

*Depth — training a model to score candidate responses by quality.*

**TL;DR:** A reward model (RM) is an LLM with a scalar head, trained on **pairwise preferences** to predict which of two responses is better. Loss is typically **Bradley-Terry** (binary log-sigmoid on the reward difference). Used as the reward signal in PPO-RLHF, as the filter in rejection sampling, and — via DPO's reparameterization — implicitly inside preference-optimization losses. Llama 3's RM **drops the Llama 2 margin term**, uses **three-way edited preferences** (chosen < edited, chosen > rejected), and **concatenates all responses per prompt into one row** at training time for efficiency. Initialization: the pretrained base model's weights + a new linear scalar head.

**Prereqs:** [_rl](_rl.md), [_rewards](_rewards.md), [dpo](dpo.md)
**Related:** [grpo](grpo.md) · [ppo](ppo.md) · [rejection-sampling](rejection-sampling.md) · [cot-reward-model](cot-reward-model.md)

---

## What it is

A **learned reward function** `r_φ(x, y) ∈ ℝ` that scores how "good" a response y is for prompt x. Used anywhere you need to rank or score responses:

- **PPO-RLHF**: the reward signal for the policy gradient.
- **Rejection sampling**: rank `K` candidates, keep top-1 (Llama 3 convention).
- **Best-of-N decoding**: sample N at inference, pick the one the RM rates highest.
- **DPO**: the reward is implicit (reparameterized into the policy), but the preference data is the same.

Architecture: take a pretrained LLM (or the current SFT checkpoint), strip the LM head, add a **linear layer → single scalar**. That scalar is the reward. Everything else is standard LLM forward.

---

## How it works

### Training data: pairwise preferences

The canonical format:

```
(prompt, chosen, rejected)
```

For each prompt, humans (or a judge model) indicate that `chosen` is a better response than `rejected`. The RM learns to assign `r_φ(x, y_chosen) > r_φ(x, y_rejected)`.

Collection methods:
- **Human annotation**: labelers rank 2–4 responses from the SFT model.
- **Judge-model generation**: a stronger model (e.g., GPT-4) ranks candidates (RLAIF).
- **Synthetic differential**: systematically generate a "better" and "worse" version (e.g., correct vs incorrect reasoning).

Llama 3 adds an **"edited" third category**: human annotators can further improve the chosen response, producing `(chosen, rejected, edited)` with preference ordering `edited > chosen > rejected`.

### Bradley-Terry (BT) loss — the standard

Model preferences via the BT model:

```
P(y_a > y_b | x) = σ(r_φ(x, y_a) - r_φ(x, y_b))
```

where σ is the logistic function. The log-likelihood loss over a preference dataset `D = {(x, y_chosen, y_rejected)}`:

```
L_BT = -E_D [ log σ(r_φ(x, y_chosen) - r_φ(x, y_rejected)) ]
```

This is the classical RLHF RM loss (InstructGPT, Llama 2, Llama 3). The Llama 2 variant added a "margin" term weighted by preference strength:

```
L_BT_margin = -E_D [ log σ(r_φ(x, y_chosen) - r_φ(x, y_rejected) - m(r_difference)) ]
```

where `m(·)` is a learned margin function based on how strongly the chosen was preferred. **Llama 3 drops this margin term** (Sec. 4.1.2), citing "diminishing improvements after data scaling."

### Implementation

Scalar head:

```python
class RewardModel(nn.Module):
    def __init__(self, base_lm):
        self.base = base_lm       # a Transformer (Llama / etc.)
        self.score_head = nn.Linear(base_lm.hidden_size, 1)

    def forward(self, tokens):
        hidden_states = self.base(tokens).last_hidden_state  # [B, S, H]
        # Take the hidden state at the LAST token (or first, depending on convention)
        last_hidden = hidden_states[:, -1, :]  # [B, H]
        reward = self.score_head(last_hidden).squeeze(-1)  # [B]
        return reward
```

Different implementations take different positions for the scalar read:
- **Last token** (Llama 3 convention): read the reward at the EOS token.
- **First token after prompt**: requires masking the prompt.
- **Average over response tokens**: less common; gives a "per-token" average reward.

For the pairwise loss:

```python
def bt_loss(rm, prompts, chosens, rejecteds):
    # Concatenate chosen and rejected sequences
    inputs_chosen   = [p + c for p, c in zip(prompts, chosens)]
    inputs_rejected = [p + r for p, r in zip(prompts, rejecteds)]

    rewards_chosen   = rm(tokenize(inputs_chosen))    # [B]
    rewards_rejected = rm(tokenize(inputs_rejected))  # [B]

    loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    return loss
```

### Llama 3's concat-and-shuffle trick (Sec. 4.1.2)

Instead of computing chosen and rejected in separate forward passes, **concatenate the prompt + all responses into a single row** with responses shuffled:

```
[prompt] [SEP] [response_1] [SEP] [response_2] [SEP] [response_3]
```

Each response's reward is read from its terminal position. Loss is applied pairwise among the responses within the row.

Llama 3 reports: "approximation to standard pairwise scoring, improves training efficiency without loss in accuracy." Pays off when you have 3+ responses per prompt (chosen, rejected, edited).

### Data filtering (Llama 3 Sec. 4.2.1)

- Labelers rate on a 4-level scale: **significantly better, better, slightly better, marginally better**.
- For RM and DPO training: **keep only "significantly better" and "better" pairs**.
- Discard "slightly better" and "marginally better" — too noisy.

### Three-way preferences

When `edited` is available, the single-row training handles all three relations:
```
edited > chosen > rejected
```

The RM should order them correctly. Loss is the sum of log-sigmoid terms over all ordered pairs (3 pairs: edited-chosen, edited-rejected, chosen-rejected).

### Training hyperparameters

Llama 3 doesn't publicly disclose RM hyperparameters, but typical defaults:
- **Initialization**: from pretrained or SFT checkpoint.
- **Optimizer**: AdamW.
- **LR**: 1e-5 to 1e-6 (lower than SFT).
- **Epochs**: 1–2 (to avoid overfitting on small preference data).
- **Batch size**: up to thousands of pairs.
- **Data scale**: 100K to 1M+ pairs (exact Llama 3 count not disclosed).

---

## Why it matters

- **The signal for RLHF.** Without an RM, PPO has nothing to optimize. The RM's quality is a hard ceiling on what PPO can achieve.
- **Filter for rejection sampling.** In Llama 3's pipeline, the RM is used to pick top-1 from K candidates. Every iteration of SFT data depends on the RM.
- **Implicit in DPO.** DPO's closed-form is equivalent to training a policy on the same BT preference data the RM would use. Understanding RMs clarifies what DPO is doing.
- **Reusable as inference-time reranker.** Train once, use it for best-of-N at deploy time to boost quality.

---

## Gotchas & tricks

- **RM overfitting is the main failure mode.** With a few thousand preference pairs, the RM can easily memorize patterns that don't generalize. 1 epoch is typical; watch validation accuracy.
- **Length bias.** RMs trained on typical preferences often prefer longer responses (a known bias in human raters). Explicit length-controlled losses or rank-biased evaluation mitigate this.
- **Calibration is poor.** Raw RM scores are not calibrated probabilities; only the differences matter. Normalize before comparing across contexts.
- **Distribution shift at RL time.** The RM is trained on SFT-distribution responses. During PPO, the policy drifts → OOD responses → RM scores become unreliable. The KL-to-reference penalty in PPO is the usual fix.
- **Ensembling helps.** Training K RMs with different seeds and averaging gives more robust rewards. Used by some frontier labs.
- **Reward hacking.** The policy will find any shortcut the RM rewards (repetition, sycophancy, specific phrases). Mitigations: KL regularization to a reference, explicit length/repetition penalties, diverse training data.
- **One RM vs multiple.** Llama 3 uses one RM across all capabilities. DeepSeek-R1 uses two (helpfulness + harmlessness, scoring different parts of the response). Llama 3 skipping the margin term → the RM trained on mixed-capability data captures "general quality," averaged.
- **Scalar head position matters.** Always take the scalar from the same positional convention (last token, etc.). If your fine-tuning data has `[EOS]` at different positions, use a `[SEP]`-like fixed token.
- **CoT Reward Models** (Kimi k1.5): generate a reasoning trace before emitting a scalar. Much higher accuracy on math / verifiable tasks. See [cot-reward-model](cot-reward-model.md).
- **Initialization from SFT vs pretrained.** SFT start gives marginally better results but costs an extra SFT round. Pretrained start is simpler; both work.
- **Don't forget the KL penalty.** An RM + PPO without a KL penalty to π_ref degenerates quickly. Always include β·KL(π_θ || π_ref).

---

## Sources

- Paper: *Training language models to follow instructions with human feedback (InstructGPT)* — Ouyang et al., 2022, arXiv 2203.02155 — canonical RM + PPO pipeline.
- Paper: *Deep Reinforcement Learning from Human Preferences* — Christiano et al., 2017, arXiv 1706.03741 — the BT-based preference loss pattern.
- Paper: *Llama 2: Open Foundation and Fine-Tuned Chat Models* — Touvron et al., 2023, arXiv 2307.09288 — the Llama 2 RM with margin term.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — drops the margin term, adds edited responses and concat-and-shuffle training.
- Paper: *Direct Preference Optimization* — Rafailov et al., 2023 — DPO reparameterizes the RM into the policy.
