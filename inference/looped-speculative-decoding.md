# DLoop: Looped Speculative Decoding

*Depth — extend the draft horizon beyond the fixed K by looping the drafter while its confidence stays above a threshold.*

**TL;DR:** Classical speculative decoding drafts exactly K tokens, verifies, repeats. DLoop keeps the drafter running **while it is still confident**, accumulates a longer draft over multiple draft steps, then verifies all accumulated tokens in a single target-model forward. The verifier only pays its cost once per loop, so wall-clock drops. Pairs with loop-aware drafter training so confidence stays calibrated across the extra drafting steps.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md)
**Related:** [mtp](../pre-training/mtp.md)

---

## What it is

Replace the fixed draft length $K$ with a confidence-gated horizon:

```
draft_tokens = []
while drafter.confidence(next_token) >= tau and len(draft_tokens) < K_max:
    tok = drafter.sample_next()
    draft_tokens.append(tok)

accepted = target.verify_in_parallel(draft_tokens)
emit(accepted); roll_back_drafter_to(last_accepted)
```

Two parameters: the confidence threshold $\tau$ and a hard cap $K_{\max}$. The expected draft length is adaptive — easy prefixes get long drafts, ambiguous positions trigger an early stop.

## How it works

### Confidence gate

For a drafter producing softmax probabilities, the confidence of the chosen token is $p = \pi_D(\hat{a} \mid s_{<t})$. DLoop stops drafting when $p < \tau$. Choosing $\tau$ well is important: too low and acceptance rate collapses; too high and draft horizons stay short.

### Loop-aware drafter training

A drafter trained only on the first-position objective loses calibration at longer draft horizons. DLoop fine-tunes the drafter to stay well-calibrated at positions 2, 3, ... of a loop; this keeps acceptance rates high across the extended horizon.

### Verifier amortisation

The expensive forward is paid once per loop, not once per token. For a loop of length $L$ with acceptance rate $\alpha$, the expected emitted tokens per target forward is $\alpha L$ — strictly higher than classical spec-dec with the same drafter.

## Why it matters

- **Lossless.** DLoop uses the same acceptance rule as classical speculative sampling, preserving the target's output distribution.
- **Composable.** Stacks on top of every current drafter: EAGLE-3, DFlash, Domino, DSpark, and MTP modules.
- **Measured gains.** 5 – 41% wall-clock speedup over the base drafter in the paper's experiments across those 5 baselines.
- The core idea — "draft longer when confident" — is a cheap change to the serving loop with no new modules.

## Gotchas & tricks

- **$\tau$ depends on the drafter.** Each base method's confidence distribution is different; sweep per-drafter.
- **Memory grows with loop length.** Longer drafts hold more KV; cap $K_{\max}$ to the batch's memory budget.
- **Confidence calibration drifts post-FT.** If you fine-tune the drafter after DLoop training, re-run the calibration sweep.
- **Interaction with batching.** Variable-length drafts complicate continuous batching; the paper reports workable schedulers but mileage varies.

## Sources

- *DLoop: Looped Speculative Decoding* — Gu, Heo, Jun, Kang, Lee, Yun, Han — NAVER AI Lab — 2026 — [arXiv:2610.07659](https://arxiv.org/abs/2610.07659)
