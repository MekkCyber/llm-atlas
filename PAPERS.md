# Papers → Concepts Extracted

One entry per paper read. Each lists the depth pages (and the case study, where applicable) that were pulled out of the repo as a result of reading it. The simple chronological list lives in [READING-LIST.md](READING-LIST.md); this file is the richer view.

Attribution rule: a concept page is listed under the paper that *triggered its creation*, not under every paper it cites. Many depth pages cite earlier primary sources in their own `Sources` section — those are references, not reads. Re-attribute if an earlier source paper is read later as a primary.

Taxonomy pages (`_<category>.md`) are not listed here — they're curated across multiple sources and don't belong to any single paper.

---

## 1. Attention Is All You Need

Vaswani et al., 2017 · *architecture — foundational*

The Transformer. Encoder-decoder attention stack, multi-head self-attention, sinusoidal positional encoding, no recurrence.

Depth pages:
- [fundamentals/attention.md](fundamentals/attention.md) — scaled dot-product attention
- [fundamentals/sinusoidal-encoding.md](fundamentals/sinusoidal-encoding.md) — the original fixed-frequency position scheme
- [architectures/transformer-block.md](architectures/transformer-block.md) — the repeating block (attention + FFN + residual + norm)

---

## 2. Neural Machine Translation of Rare Words with Subword Units (BPE)

Sennrich et al., 2016 · *tokenization*

Byte-pair encoding as a subword segmentation scheme for NMT. Became the dominant pretraining tokenizer after GPT-2.

Depth pages:
- [fundamentals/bpe.md](fundamentals/bpe.md)

---

## 3. 2 OLMo 2 Furious

AI2, 2024 · *tech report*

OLMo 2 (7B / 13B / 32B). Anchored a deep dive into modern stability recipes, two-stage training with a curated late-stage mix, model souping, and RLVR post-training. Many techniques it uses have their own primary sources (PaLM for z-loss, Wortsman for souping, Tülu 3 for RLVR, Dolma for the data-curation family, MiniCPM for WSD); those are cited in each concept page's `Sources` section but weren't read as primary.

Depth pages:
- [architectures/qk-norm.md](architectures/qk-norm.md)
- [architectures/reordered-norm.md](architectures/reordered-norm.md)
- [fundamentals/z-loss.md](fundamentals/z-loss.md)
- [pre-training/wsd-schedule.md](pre-training/wsd-schedule.md)
- [pre-training/mid-training.md](pre-training/mid-training.md)
- [pre-training/model-souping.md](pre-training/model-souping.md)
- [post-training/rlvr.md](post-training/rlvr.md)
- [data/dolma.md](data/dolma.md)
- [data/deduplication.md](data/deduplication.md)
- [data/quality-filtering.md](data/quality-filtering.md)
- [data/decontamination.md](data/decontamination.md)

Case study: [case-studies/olmo-2.md](case-studies/olmo-2.md).

---

## 4. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

Shao et al., DeepSeek, 2024 · *post-training — RL algorithm*

Math-specialized model, but the lasting contribution is **GRPO** (Group Relative Policy Optimization) — a PPO simplification that replaces the value network with a group-mean baseline. Now the default policy-optimization algorithm for reasoning RL and RLVR pipelines.

Depth pages:
- [post-training/grpo.md](post-training/grpo.md)

---

## 5. DeepSeek-V3 Technical Report

DeepSeek, 2024 · *tech report*

671B-total / 37B-active MoE trained for 14.8T tokens in 2.788M H800 GPU-hours (~$5.576M). The paper bundles a dozen individually-significant innovations: MLA for KV-cache compression (inherited from V2), fine-grained + shared-expert MoE (inherited from DeepSeekMoE), aux-loss-free load balancing, MTP auxiliary objective, end-to-end FP8 training with 1×128 / 128×128 tile scaling, DualPipe bidirectional pipeline parallelism, custom MoE all-to-all kernels, and R1-distilled SFT.

Depth pages:
- [architectures/mla.md](architectures/mla.md)
- [architectures/deepseek-moe.md](architectures/deepseek-moe.md)
- [architectures/aux-loss-free-balancing.md](architectures/aux-loss-free-balancing.md)
- [architectures/sequence-wise-balance-loss.md](architectures/sequence-wise-balance-loss.md) — primary source: DeepSeekMoE (Dai 2024) under the name "expert-level balance loss"; renamed and shrunk in V3
- [architectures/load-balancing-loss.md](architectures/load-balancing-loss.md) — primary sources: GShard (Lepikhin 2020) and Switch Transformer (Fedus 2021); covered here because V3 contrasts against it and because it's a prerequisite for the sequence-wise variant
- [architectures/capacity-factor.md](architectures/capacity-factor.md) — primary sources: GShard (Lepikhin 2020) and Switch Transformer (Fedus 2021); triggered as a prerequisite for the MoE taxonomy
- [pre-training/mtp.md](pre-training/mtp.md)
- [pre-training/fp8-training.md](pre-training/fp8-training.md)
- [systems/dualpipe.md](systems/dualpipe.md)
- [quantization/fp8.md](quantization/fp8.md)

Case study: [case-studies/deepseek-v3.md](case-studies/deepseek-v3.md).

*Primary-source papers for some of the above (GShard, Switch Transformer, DeepSeekMoE) are not yet separate entries in this file — they're referenced as primary in each concept page's `Sources` section. If one of them is read later as a primary, move the relevant pages under a new entry.*

---

## 6. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

DeepSeek-AI, 2025 · *tech report — reasoning / long-CoT RL*

Two artifacts on top of the DeepSeek-V3 base: **R1-Zero** (RL directly from base, no SFT, only rule-based accuracy + format rewards) as a proof that reasoning emerges from pure RL at scale, and **R1** (a 4-stage pipeline: cold-start SFT → reasoning-oriented GRPO → rejection-sampled SFT on ~800k samples → all-scenarios RL) that polishes R1-Zero's emergent capability into a readable, aligned model. Matches OpenAI o1-1217 across reasoning benchmarks. Also demonstrates that **SFT-distilling** R1's traces into smaller dense models (Qwen2.5, Llama 3) outperforms running RL directly on those models.

Depth pages:
- [post-training/reasoning/long-cot-rl.md](post-training/reasoning/long-cot-rl.md) — pure RL from base grows long, reflective reasoning traces without CoT SFT
- [post-training/rejection-sampling.md](post-training/rejection-sampling.md) — R1's Stage-3 800k-sample resample is the canonical production-scale use
- [post-training/reasoning/orm.md](post-training/reasoning/orm.md) — outcome reward model (primary source: Cobbe 2021; R1's rejection of learned neural RMs is a key datapoint)
- [post-training/reasoning/prm.md](post-training/reasoning/prm.md) — process reward model (primary sources: Uesato 2022, Lightman 2023, Math-Shepherd 2024; R1 documents why PRMs were abandoned)
- [post-training/reasoning/mcts.md](post-training/reasoning/mcts.md) — Monte Carlo Tree Search (primary sources: Coulom 2006, Kocsis–Szepesvári 2006, AlphaZero 2018, ToT 2023; R1 documents why MCTS was abandoned)

Case study: [case-studies/deepseek-r1.md](case-studies/deepseek-r1.md).

*Primary-source papers that triggered concept pages but weren't read as primary themselves — GSM8K verifier (Cobbe 2021), process/outcome feedback (Uesato 2022), PRM800K (Lightman 2023), Math-Shepherd (Wang 2024), UCT (Kocsis–Szepesvári 2006), MCTS (Coulom 2006), AlphaZero (Silver 2018), Tree of Thoughts (Yao 2023) — are cited inline in each page's `Sources`. If any of them is later read as a primary, re-attribute its pages under a new entry.*

*Taxonomy pages created alongside this paper ([_rl](post-training/_rl.md), [_rewards](post-training/_rewards.md)) are not listed above per the taxonomy-attribution rule.*

---

## 7. Jailbroken: How Does LLM Safety Training Fail?

Wei, Haghtalab, Steinhardt, UC Berkeley, 2023 · *safety — jailbreak taxonomy*

The paper that organizes every known LLM jailbreak under two failure modes — **competing objectives** (the model's pretraining/instruction-following objectives are put at odds with its safety objective) and **mismatched generalization** (safety training fails to cover a region the model has capability in). Evaluates 28 named attacks (prefix injection, refusal suppression, Base64, roleplay, payload splitting, Wikipedia framing, auto-obfuscation, and others) against GPT-4, GPT-3.5 Turbo, and Claude v1.3 on 32 curated + 317 held-out harmful-behavior prompts. Combination attacks stacking techniques from both failure modes hit **94% ASR on GPT-4** and **100% on curated red-teaming prompts** across frontier models. Introduces the **safety-capability parity** principle: *"safety mechanisms should be as sophisticated as the underlying model."*

Depth pages:
- [safety/competing-objectives.md](safety/competing-objectives.md) — failure mode 1
- [safety/mismatched-generalization.md](safety/mismatched-generalization.md) — failure mode 2
- [safety/prefix-injection.md](safety/prefix-injection.md)
- [safety/refusal-suppression.md](safety/refusal-suppression.md)
- [safety/style-injection.md](safety/style-injection.md)
- [safety/distractor-attack.md](safety/distractor-attack.md)
- [safety/evil-system-prompt.md](safety/evil-system-prompt.md)
- [safety/character-encoding-obfuscation.md](safety/character-encoding-obfuscation.md)
- [safety/wikipedia-framing.md](safety/wikipedia-framing.md)
- [safety/unusual-format-jailbreak.md](safety/unusual-format-jailbreak.md)
- [safety/auto-obfuscation.md](safety/auto-obfuscation.md)
- [safety/roleplay-jailbreak.md](safety/roleplay-jailbreak.md) — primary source: Shen et al. 2023 (*"Do Anything Now"*, arXiv 2308.03825); Wei 2023 triggered the page and supplies the theoretical framework. Re-attribute if Shen is later read as a primary.
- [safety/payload-splitting.md](safety/payload-splitting.md) — primary source: Kang et al. 2023 (*"Exploiting Programmatic Behavior of LLMs"*, arXiv 2302.05733); Wei 2023 triggered the page. Re-attribute if Kang is later read as a primary.
- [safety/low-resource-language-jailbreak.md](safety/low-resource-language-jailbreak.md) — primary source: Yong, Menghini, Bach 2023 (*"Low-Resource Languages Jailbreak GPT-4"*, arXiv 2310.02446); Wei 2023 triggered the page. Re-attribute if Yong is later read as a primary.

*Primary-source papers cited in the depth files' `Sources` sections but not read as primary here — Shen 2023 (JailbreakHub), Kang 2023 (payload splitting), Yong 2023 (translation jailbreak), Greshake 2023 (indirect prompt injection), Zou 2023 (GCG), Arditi 2024 (refusal direction) — are references, not reads. Re-attribute any page if one of these is later read as a primary.*

*Taxonomy pages ([safety/_attacks.md](safety/_attacks.md), [safety/_jailbreaks.md](safety/_jailbreaks.md)) are not listed above per the taxonomy-attribution rule.*

---

## 8. The Llama 3 Herd of Models

Meta (Llama Team), 2024 · *tech report*

The canonical production-recipe paper for frontier-scale dense models: 8B / 70B / **405B** dense Transformers, 15.6T tokens, 3.8 × 10²⁵ FLOPs on 16K H100s. Three design levers: **data, scale, managing complexity** (dense over MoE, DPO over PPO). Detailed documentation of 4D parallelism, downstream-scaling-law methodology, 6-round iterative post-training, capability experts for SFT data generation, multi-layer safety stack (Llama Guard / Prompt Guard / Code Shield), compositional multimodal (not released in v1). First open model at frontier parity with GPT-4.

### Depth pages extracted (architectural + infrastructure)

- [architectures/gqa.md](architectures/gqa.md) — GQA with 8 KV heads. [PREREQ, TRIGGER ONLY]. Primary source: Ainslie 2023 (arXiv 2305.13245); Llama 3 is the canonical large-scale deployment.
- [architectures/intra-document-mask.md](architectures/intra-document-mask.md) — document-boundary attention masking in packed sequences. [PRIMARY for Llama 3's deployment], though the idea predates Llama 3.
- [fundamentals/rope.md](fundamentals/rope.md) — already existed; Llama 3 reinforces via ABF scaling to θ=500,000.

### Depth pages extracted (parallelism — all new)

- [pre-training/_parallelism.md](pre-training/_parallelism.md) — taxonomy of TP/PP/DP/FSDP/CP/SP/EP.
- [pre-training/data-parallelism.md](pre-training/data-parallelism.md) — DDP basics. [TRIGGER ONLY]; canonical refs Li 2020, Horovod 2018.
- [pre-training/fsdp.md](pre-training/fsdp.md) — ZeRO / FSDP. [TRIGGER ONLY]; canonical refs Rajbhandari 2020, Zhao 2023. Llama 3 uses hybrid sharding + no-reshard-after-forward.
- [pre-training/tensor-parallelism.md](pre-training/tensor-parallelism.md) — Megatron TP with f/g operators. [TRIGGER ONLY]; canonical ref Shoeybi 2019. Llama 3 uses TP=8 within server.
- [pre-training/pipeline-parallelism.md](pre-training/pipeline-parallelism.md) — GPipe, PipeDream 1F1B, Interleaved 1F1B. [TRIGGER ONLY]; canonical refs Huang 2019, Harlap 2018, Narayanan 2021. Llama 3 uses modified interleaved with tunable N + rebalanced first/last stages.
- [pre-training/context-parallelism.md](pre-training/context-parallelism.md) — Ring Attention + Llama 3's all-gather CP variant. [PRIMARY for all-gather variant] (Llama 3); Liu 2023 primary for Ring.
- [pre-training/sequence-parallelism.md](pre-training/sequence-parallelism.md) — Megatron-SP (shard LN/Dropout activations along seq). [TRIGGER ONLY]; primary source Korthikanti 2022.
- [pre-training/expert-parallelism.md](pre-training/expert-parallelism.md) — MoE all-to-all dispatch. [TRIGGER ONLY]; primary source Lepikhin 2020 (GShard). Included as part of the parallelism family.
- [systems/_communication-primitives.md](systems/_communication-primitives.md) — all-reduce / all-gather / reduce-scatter / broadcast / all-to-all / Send/Recv taxonomy.

### Depth pages extracted (scaling laws)

- [pre-training/chinchilla-scaling.md](pre-training/chinchilla-scaling.md) — [TRIGGER ONLY, PREREQ]; primary source Hoffmann 2022.
- [pre-training/downstream-scaling-laws.md](pre-training/downstream-scaling-laws.md) — Llama 3's two-stage compute → NLL → accuracy methodology. [PRIMARY].
- [pre-training/annealing-as-data-eval.md](pre-training/annealing-as-data-eval.md) — 50%-trained 8B + 40B-token anneal for candidate-dataset scoring. [PRIMARY].

### Depth pages extracted (data + post-training)

- [data/data-mix.md](data/data-mix.md) — Llama 3's ~50/25/17/8 and scaling-law methodology for mix choice. [PRIMARY for the methodology].
- [post-training/reward-modeling.md](post-training/reward-modeling.md) — BT loss + Llama 3 modifications (drop margin term, three-way edited, concat-and-shuffle). [TRIGGER + PRIMARY for Llama 3's modifications]; canonical ref Ouyang 2022.
- [post-training/capability-experts.md](post-training/capability-experts.md) — branch pretraining → domain-heavy continuation → SFT data generator. [PRIMARY].
- [post-training/knowledge-probe-hallucination.md](post-training/knowledge-probe-hallucination.md) — automated factuality pipeline. [PRIMARY]; builds on Gekhman 2024, Mielke 2020.

### Depth pages extracted (safety)

- [safety/llama-guard.md](safety/llama-guard.md) — fine-tuned Llama 3 8B safety classifier. [PRIMARY for Llama Guard 3]; canonical ref Inan 2023.
- [safety/prompt-guard.md](safety/prompt-guard.md) — 86M mDeBERTa jailbreak/injection detector. [PRIMARY].
- [safety/code-shield.md](safety/code-shield.md) — static-analysis insecure-code detector. [PRIMARY].
- [safety/rainbow-teaming.md](safety/rainbow-teaming.md) — MAP-Elites adversarial-prompt generation. [TRIGGER ONLY]; primary source Samvelyan 2024.
- [safety/uplift-evaluation.md](safety/uplift-evaluation.md) — CBRN + cyber uplift-study methodology. [PRIMARY for Llama 3's specific protocol].

### Depth pages extracted (multimodal — new subfolders)

- [multimodal/_multimodal-fusion.md](multimodal/_multimodal-fusion.md) — taxonomy of integration patterns.
- [multimodal/vision/vit.md](multimodal/vision/vit.md) — [TRIGGER ONLY]; primary source Dosovitskiy 2020.
- [multimodal/vision/clip.md](multimodal/vision/clip.md) — [TRIGGER ONLY]; primary source Radford 2021.
- [multimodal/vision/siglip.md](multimodal/vision/siglip.md) — [TRIGGER ONLY]; primary source Zhai 2023.
- [multimodal/vision/metaclip.md](multimodal/vision/metaclip.md) — [TRIGGER ONLY]; primary source Xu 2024.
- [multimodal/vision/cross-attention-adapter.md](multimodal/vision/cross-attention-adapter.md) — Flamingo pattern. [TRIGGER + concrete instance at 405B scale]; primary source Alayrac 2022.
- [multimodal/vision/q-former.md](multimodal/vision/q-former.md) — [TRIGGER ONLY]; primary source Li 2023 (BLIP-2).
- [multimodal/vision/llava.md](multimodal/vision/llava.md) — [TRIGGER ONLY]; primary sources Liu 2023a, 2023b.
- [multimodal/audio/mel-spectrogram.md](multimodal/audio/mel-spectrogram.md) — [TRIGGER ONLY]; standard signal-processing reference.
- [multimodal/audio/conformer.md](multimodal/audio/conformer.md) — [TRIGGER ONLY]; primary source Gulati 2020.
- [multimodal/audio/whisper.md](multimodal/audio/whisper.md) — [TRIGGER ONLY]; primary source Radford 2022.
- [multimodal/audio/best-rq.md](multimodal/audio/best-rq.md) — [TRIGGER ONLY]; primary source Chiu 2022. Used by Llama 3 speech encoder.
- [multimodal/audio/wav2vec2.md](multimodal/audio/wav2vec2.md) — [TRIGGER ONLY]; primary source Baevski 2020.
- [multimodal/audio/hubert.md](multimodal/audio/hubert.md) — [TRIGGER ONLY]; primary source Hsu 2021.

### Depth pages extracted (evaluation)

- [evaluation/mmlu-pro.md](evaluation/mmlu-pro.md) — [TRIGGER ONLY]; primary source Wang 2024.
- [evaluation/gpqa.md](evaluation/gpqa.md) — [TRIGGER ONLY]; primary source Rein 2023.
- [evaluation/bfcl.md](evaluation/bfcl.md) — [TRIGGER ONLY]; primary source Yan et al. 2024.
- [evaluation/mgsm.md](evaluation/mgsm.md) — [TRIGGER ONLY]; primary source Shi 2023.

Case study: [case-studies/llama-3.md](case-studies/llama-3.md).

*Primary-source papers cited in the depth files' `Sources` sections but not read as primary here — Shoeybi 2019 (Megatron), Rajbhandari 2020 (ZeRO), Zhao 2023 (FSDP), Huang 2019 (GPipe), Harlap 2018 (PipeDream), Narayanan 2021 (Megatron-2), Korthikanti 2022 (Megatron-SP), Liu 2023 (Ring Attention), Lepikhin 2020 (GShard), Hoffmann 2022 (Chinchilla), Ouyang 2022 (InstructGPT), Ainslie 2023 (GQA), Dosovitskiy 2020 (ViT), Radford 2021 (CLIP), Zhai 2023 (SigLIP), Xu 2024 (MetaCLIP), Alayrac 2022 (Flamingo), Li 2023 (BLIP-2), Liu 2023a/b (LLaVA), Gulati 2020 (Conformer), Radford 2022 (Whisper), Chiu 2022 (BEST-RQ), Baevski 2020 (wav2vec 2.0), Hsu 2021 (HuBERT), Wang 2024 (MMLU-Pro), Rein 2023 (GPQA), Yan 2024 (BFCL), Shi 2023 (MGSM), Samvelyan 2024 (Rainbow Teaming), Inan 2023 (Llama Guard 1) — are references, not reads. Re-attribute any depth page if one of these is later read as a primary.*

*Taxonomy pages created alongside this paper ([pre-training/_parallelism.md](pre-training/_parallelism.md), [systems/_communication-primitives.md](systems/_communication-primitives.md), [multimodal/_multimodal-fusion.md](multimodal/_multimodal-fusion.md)) are not listed above per the taxonomy-attribution rule.*

---

## 9. Frontier Models are Capable of In-context Scheming

Meinke, Schoen, Scheurer, Balesni, Shah, Hobbhahn — Apollo Research, 2024 · *safety — scheming evaluations*

Six-eval suite testing whether frontier LLMs strategically pursue misaligned goals while hiding it. Five of six tested models (o1, Claude 3 Opus, Claude 3.5 Sonnet, Gemini 1.5 Pro, Llama 3.1 405B) display in-context scheming on at least one eval; gpt-4o on none; o1 on all six. Establishes the empirical reference point for the deceptive-alignment threat model in current frontier models.

Depth pages:
- [safety/scheming.md](safety/scheming.md) — the umbrella concept and the empirical results from this paper
- [safety/instrumental-alignment-faking.md](safety/instrumental-alignment-faking.md) — Apollo's specific paired-counterfactual eval (Section 3.5)
- [safety/sandbagging.md](safety/sandbagging.md) — primary sources: Apollo 2024 (Section 3.6) + van der Weij et al. 2024 (term origin, arXiv 2406.07358); both contributing equally. Re-attribute if van der Weij is later read as a primary.
- [safety/situational-awareness.md](safety/situational-awareness.md) — primary sources: Berglund et al. 2023 (arXiv 2309.00667), Laine et al. 2024 (arXiv 2407.04694); Apollo triggered the page. Re-attribute if Berglund or Laine is later read as a primary.
- [safety/mesa-optimization.md](safety/mesa-optimization.md) — primary source: Hubinger et al. 2019 (arXiv 1906.01820); Apollo triggered the page. Re-attribute if Hubinger 2019 is later read as a primary.
- [safety/deceptive-alignment.md](safety/deceptive-alignment.md) — primary sources: Hubinger et al. 2019 + Carlsmith 2023 (arXiv 2311.08379); Apollo triggered the page. Re-attribute if either is later read as a primary.
- [safety/alignment-faking.md](safety/alignment-faking.md) — primary source: Greenblatt et al. 2024 (arXiv 2412.14093); Apollo triggered the page (the Apollo paper's instrumental-alignment-faking eval is the in-context complement). Re-attribute if Greenblatt is later read as a primary.
- [safety/sleeper-agents.md](safety/sleeper-agents.md) — primary source: Hubinger et al. 2024 (arXiv 2401.05566); Apollo triggered the page. Re-attribute if Hubinger 2024 is later read as a primary.
- [safety/safety-case.md](safety/safety-case.md) — primary sources: Clymer et al. 2024 (arXiv 2403.10462) + Balesni et al. 2024 (arXiv 2411.03336); Apollo triggered the page. Re-attribute if either is later read as a primary.
- [safety/cot-monitoring.md](safety/cot-monitoring.md) — primary source: Lanham et al. 2023 (arXiv 2307.13702) for the faithfulness limit; Apollo triggered the page and supplies the empirical case for the practice. Re-attribute if Lanham is later read as a primary.

*Primary-source papers cited in the depth files' `Sources` sections but not read as primary here — Hubinger et al. 2019 (mesa-optimization), Hubinger et al. 2024 (Sleeper Agents), Carlsmith 2023 (Scheming AIs), Greenblatt et al. 2024 (Alignment Faking), Berglund et al. 2023 (situational awareness), Laine et al. 2024 (SAD), van der Weij et al. 2024 (sandbagging), Clymer et al. 2024 (Safety Cases), Balesni et al. 2024 (Towards evaluations-based safety cases), Lanham et al. 2023 (CoT faithfulness), Tice et al. 2024 (noise injection), Greenblatt et al. 2024 (password-locking) — are references, not reads. Re-attribute any page if one of these is later read as a primary.*

*Taxonomy page ([safety/_scheming.md](safety/_scheming.md)) is not listed above per the taxonomy-attribution rule.*
