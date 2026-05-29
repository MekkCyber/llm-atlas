# Relevance filter — keep vs skip

The digest casts a wide net for anything that touches *how modern generative AI is trained, served, evaluated, or kept safe*. The bar is **topical fit**, not novelty. A small incremental advance in a kept topic is more useful to the graph than a flashy result in an off-topic field.

## KEEP — papers in scope

| Topic | Example keywords (title/abstract triggers) |
| --- | --- |
| **LLM training & post-training** | pretraining, scaling laws, base model, instruction tuning, SFT, RLHF, PPO, GRPO, DPO, RLVR, reward modeling, rollouts, preference learning, KTO, ORPO, IPO |
| **Reasoning** | long chain-of-thought, CoT, reasoning RL, self-consistency, tree search, MCTS, process reward, verifier, math/code reasoning, R1-style, o1-style |
| **Multimodal LLMs** | vision-language, VLM, image+text, video understanding, audio-language, projection layer, vision tokenizer, any-to-any |
| **Diffusion (generative)** | diffusion model, flow matching, rectified flow, consistency model, score-based, latent diffusion, image/video/audio generation, DiT, UNet |
| **Architectures** | mixture of experts, MoE, attention variant, MLA, GQA, MQA, RoPE, ALiBi, SSM, Mamba, state-space, hybrid attention, sparse attention, linear attention, sliding window |
| **Efficient inference** | KV cache, paged attention, continuous batching, speculative decoding, draft model, EAGLE, Medusa, vLLM, SGLang, prefill/decode, disaggregated serving |
| **Quantization** | FP8, FP4, MXFP4, INT4, INT8, GPTQ, AWQ, SmoothQuant, calibration, mixed precision, weight-only quantization, activation quantization |
| **Distributed training systems** | FSDP, ZeRO, tensor parallel, pipeline parallel, expert parallel, sequence parallel, Ring attention, DualPipe, all-to-all, collective communication, fault tolerance |
| **Data** | data curation, deduplication, quality filtering, decontamination, synthetic data, data mixture, scaling data, web-scale, code data, multilingual data |
| **Agents / tool use** | tool calling, function calling, agent, MCP, multi-step, multi-turn, environment interaction, computer use, browser use, training agents with RL |
| **Safety & alignment** | RLHF for safety, constitutional AI, red-teaming, jailbreak, refusal, alignment faking, scheming, deceptive alignment, dangerous capability, oversight, scalable oversight |
| **Interpretability** | mechanistic interpretability, circuits, SAE, sparse autoencoder, probing, activation steering, logit lens, feature visualization, monosemanticity |
| **Evaluation** | benchmark for LLMs/VLMs/agents, contamination, pairwise comparison, reward model eval, judge LLM, capability evaluation, dangerous capability eval |

## SKIP — out of scope

| Topic | Example keywords |
| --- | --- |
| **Pure computer vision (non-generative)** | image classification, object detection, segmentation, depth estimation, optical flow — *unless* it's a vision encoder for a VLM |
| **Classical NLP without LLM angle** | parsing, NER, sentiment analysis, classical IR, classical MT — *unless* the method is LLM-based or about training data for LLMs |
| **Robotics-only** | robot manipulation, locomotion, classical RL on physical robots — *unless* the paper proposes a foundation model / VLA |
| **Theory disconnected from neural nets** | pure learning theory, classical statistics, kernel methods, classical optimization on non-NN settings |
| **Application papers without method contribution** | "we used GPT-4 for X in domain Y" papers that don't propose a new training/inference/eval/safety method |
| **Hardware-only** | chip-design papers without a software/training contribution |
| **Audio (non-LLM)** | classical speech recognition, classical TTS — *unless* it's an LLM-based audio model |
| **Bio/chem ML** | protein structure, molecular property prediction — *unless* the contribution is a method transferrable to LLMs |
| **Federated / privacy / FL** | unless the method is interesting for LLM-scale training |
| **Adversarial robustness on small image classifiers** | classical adv-examples on ImageNet-scale CNNs |

## Ambiguous → keep if X

| Borderline | Keep if … |
| --- | --- |
| RL paper not obviously about LLMs | the algorithm is plausibly applicable to LLM post-training (PPO/GRPO variants, off-policy methods, reward modeling). |
| Diffusion paper for images only | it introduces a training/sampling technique transferrable to text/video diffusion or to general generative modeling. |
| Generic optimization paper | it's evaluated on transformer training or proposes something for large-batch / large-scale training. |
| Vision encoder paper | the encoder is positioned for use in a multimodal LLM. |
| Robotics + foundation model | the paper trains or fine-tunes a VLA / foundation model — keep. Pure controller — skip. |
| Survey / position paper | it covers a kept topic and could ground a new taxonomy file. |
| Benchmark introduction | it benchmarks LLMs, VLMs, agents, or safety capabilities. Tiny domain-specific benchmarks → skip unless they fill a gap. |
| Application paper | it proposes a new training, prompting, or eval *method* (not just an off-the-shelf application). |

## How to apply the filter

For each candidate paper:

1. Skim the title and the first sentence of the abstract.
2. If it clearly hits a KEEP keyword → keep.
3. If it clearly hits a SKIP keyword and nothing else → skip.
4. Otherwise → check the ambiguous table. **Default to keep** when uncertain — better a one-paragraph "borderline, here's why I included it" section than a missed paper.

When skipping borderline papers, you can omit them silently. When *including* a borderline paper, add a short italic note in the section: *"Borderline: included because <reason>."*

## Volume guidance

No fixed cap. A typical HF Daily Papers page has 5–20 entries; expect to keep 60–80% on a busy day. If you find yourself keeping 100% with no skips, you're probably being too generous — re-check the SKIP column.
