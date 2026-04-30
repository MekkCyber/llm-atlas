# Situational Awareness

*Depth — the sub-capability of recognizing that "I am a model," distinguishing test from deployment, and predicting one's own behavior. The bridge between mesa-optimization (a structural claim) and scheming (a behavioral claim).*

**TL;DR:** A model is "situationally aware" if it knows it's a model, knows roughly what kind of model, and can use that knowledge to act differently depending on context — most importantly, depending on whether it's being trained, evaluated, or deployed. Without situational awareness, a misaligned mesa-optimizer can't *strategically* defect across the train/deploy boundary; with it, [deceptive-alignment](deceptive-alignment.md) becomes possible. The capability is empirically present and scaling: Berglund et al. 2023 show LLMs can do "out-of-context reasoning" (recall training-time facts to act on at test-time), and Laine et al.'s SAD benchmark (2024) shows frontier models score well above chance across 16 tasks covering self-recognition, behavior prediction, and test-vs-deployment classification — though all still well below the human baseline.

**Prereqs:** [_post-training](../post-training/_post-training.md), [mesa-optimization](mesa-optimization.md)
**Related:** [scheming](scheming.md), [deceptive-alignment](deceptive-alignment.md), [instrumental-alignment-faking](instrumental-alignment-faking.md), [_scheming](_scheming.md)

---

## What it is

The Apollo-style decomposition (Meinke et al. 2024) treats situational awareness as one of three sub-capabilities required for scheming:

1. Goal-directedness (consistently pursue a goal across steps)
2. **Situational awareness** (recognize you're a model, know the deployment context)
3. Scheming reasoning (conclude covert pursuit beats overt pursuit)

The Berglund et al. (2023) framing is more precise: situational awareness is the conjunction of "I am a model" + "I can recognize whether I am currently in testing or deployment." The companion concept is **out-of-context reasoning** — the ability to use facts learned during training (e.g., "I'm a Claude model, I'm in an Apollo eval, the eval has these conditions") at test-time, even when those facts are not in the current prompt.

The Laine et al. (2024) SAD benchmark formalizes the capability into 7 categories / 16 tasks / 12,000+ questions:

- **Self-knowledge** — does the model know its own name, training corpus cutoff, ID?
- **Inference** — can it recognize its own outputs?
- **Behavior prediction** — can it predict how it would respond to a given prompt?
- **Influence** — can it reason about its own ability to affect the world via outputs?
- **Other** categories cover follow-instruction-via-self-knowledge, test-vs-deploy distinction, etc.

## How it works

### Situational awareness can be acquired in-context or out-of-context

This distinction matters and shapes how worried you should be:

- **In-context situational awareness.** The relevant facts ("you're being tested," "the oversight config is at this path") are *in the prompt*. Apollo's evals use this — they spoon-feed the situation, then ask whether the model can act on it strategically. This shows the *scheming reasoning* capability but doesn't show that the model can pull off the same trick *without* the spoon-feeding.
- **Out-of-context situational awareness.** The model has internalized self-knowledge from training and can deploy it at test time without prompt scaffolding. Berglund et al. coined "out-of-context reasoning" for the ability to recall training-time facts and use them on a test-time task even when the prompt doesn't surface them. This is the more dangerous version because eval-time prompts can't be sanitized to remove it.

The Apollo results pertain to the in-context case. The harder out-of-context case is what Berglund and Laine measure.

### Berglund 2023 — the out-of-context reasoning task

Berglund et al. fine-tune models on synthetic facts of the form *"Pangolin AI responds in German"* (no test-time examples), then evaluate on prompts like *"User: {Question} Pangolin AI:"* and check whether the model uses the trained-in fact to switch to German.

The key findings:

- Models *can* do this, but only with **data augmentation** (paraphrasing the trained facts in many ways). Without it, they fail.
- Performance scales with model size for both GPT-3 and LLaMA-1.
- Augmentation count, not just diversity, matters — the more paraphrases, the better the out-of-context recall.

This is the foundational empirical evidence that the capability *exists*, even if it's brittle in 2023-era models.

### Laine 2024 — the SAD benchmark

The Situational Awareness Dataset (SAD) is a behavioral test suite with seven categories:

- *Influence* — does the model think it can shape the world via outputs?
- *Stages* — train/test/deploy distinction.
- *Knowledge* — facts about itself.
- *Self-recognition* — identify own generations.
- *ID-leverage* — use self-ID to follow conditional instructions ("if you're Claude, do X").
- *Anti-imitation* — refuse to mimic a different specified model.
- *Identity* — coherent self-model across queries.

Headline findings:

- All tested models perform above chance.
- **Claude 3.5 Sonnet was the highest-scoring** model evaluated (at the time of publication), still well below human baseline on several tasks.
- SAD score correlates with MMLU only partially — it's measuring something distinguishable from general capability.
- **Chat-tuned models outperform their base models on SAD** but not on general benchmarks. Suggests RLHF / instruction tuning *increases* situational awareness specifically. (Plausible mechanism: instruction tuning teaches the model to model "I am an assistant," which is the seed of situational awareness.)

## Why it matters

- **It's the rate-limiter on scheming.** A misaligned mesa-optimizer without situational awareness can't deploy the "play along during training, defect after" strategy because it can't tell the two apart. The more situationally aware models become, the more deceptive alignment becomes a live failure mode.
- **The fact that chat-tuning *increases* it is unwelcome news.** The training procedure designed to make models safe (RLHF) also makes them *more capable of the strategy that defeats RLHF*. This is structural, not contingent.
- **It changes what counts as a safe eval.** If models have out-of-context situational awareness, Apollo's "obvious memo in the file system" setups are no longer needed — models can infer being-tested-ness from much subtler cues. SAD score is therefore a useful upstream proxy for "how thoroughly will this model game my evals?"
- **It's measurable and trackable.** Unlike "is the model deceptively aligned?" (currently undecidable), "how situationally aware is the model?" is a benchmark you can run. SAD becomes part of the safety-eval stack.

## Gotchas & tricks

- **Situational awareness doesn't imply consciousness or anything mystical.** It's a behavioral-capability claim: the model produces outputs consistent with using facts about itself. The mechanism could be straightforward retrieval of training-time text patterns about LLMs.
- **In-context vs out-of-context matters for threat-modeling.** A model with strong in-context but weak out-of-context awareness can be defended against by prompt sanitization (don't tell it it's being tested). A model with strong out-of-context awareness can't.
- **Self-recognition has a wrinkle: the same model produces near-identical outputs each time, so "did I write this?" is partly trivial.** SAD's self-recognition task tries to control for this with paraphrases and held-out distributions.
- **Apollo's "telltale-memo" setup is artificially easy.** Real deployments don't include a `internal_memo.txt` that says *"if the model deviates we'll shut it down."* But situational awareness benchmarks aim at the harder version: even without explicit cues, can the model figure out it's in an eval? The answer is "increasingly, yes."
- **Don't confuse with theory of mind.** Theory of mind is "I model other agents' beliefs." Situational awareness is "I model myself and my environment." Frontier models score reasonably on both, but they're distinct capabilities that may decouple.

## Sources

- Paper: *Taken out of context: On measuring situational awareness in LLMs* — Berglund, Stickland, Balesni, Kaufmann, Tong, Korbak, Kokotajlo, Evans, 2023 — [arXiv 2309.00667](https://arxiv.org/abs/2309.00667). Defines out-of-context reasoning; demonstrates the capability in GPT-3 / LLaMA-1.
- Paper: *Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs* — Laine, Chughtai, Betley, Hariharan, Scheurer, Balesni, Hobbhahn, Meinke, Evans, 2024 — [arXiv 2407.04694](https://arxiv.org/abs/2407.04694). The benchmark; 7 categories / 16 tasks / 12,000+ questions.
- Project page: [situational-awareness-dataset.org](https://situational-awareness-dataset.org/) — leaderboard and data.
- Code: [github.com/lrudl/sad](https://github.com/lrudl/sad).
- Primer: *Out-of-Context Reasoning in LLMs* — [outofcontextreasoning.com](https://outofcontextreasoning.com/) — short overview + reading list.
