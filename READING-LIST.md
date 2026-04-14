# Reading List — 100 Papers

A curated zero-to-hero reading list across every section of this repo. Order within each section is roughly historical / pedagogical — earlier papers usually make later ones easier.

Check items off as you extract concepts from them into the relevant topic folders.

---

## Fundamentals (10)

- [x] 1. **Attention Is All You Need** — Vaswani et al., 2017 — the Transformer.
- [x] 2. **Neural Machine Translation of Rare Words with Subword Units (BPE)** — Sennrich et al., 2016 — subword tokenization.
- [ ] 3. **SentencePiece** — Kudo & Richardson, 2018 — language-agnostic tokenization.
- [ ] 4. **Adam: A Method for Stochastic Optimization** — Kingma & Ba, 2014.
- [ ] 5. **Decoupled Weight Decay Regularization (AdamW)** — Loshchilov & Hutter, 2017.
- [ ] 6. **Layer Normalization** — Ba et al., 2016.
- [ ] 7. **Root Mean Square Layer Normalization (RMSNorm)** — Zhang & Sennrich, 2019.
- [ ] 8. **Gaussian Error Linear Units (GELU)** — Hendrycks & Gimpel, 2016.
- [ ] 9. **GLU Variants Improve Transformer (SwiGLU)** — Shazeer, 2020.
- [ ] 10. **RoFormer: Rotary Position Embedding (RoPE)** — Su et al., 2021.

## Architectures (12)

- [ ] 11. **Improving Language Understanding by Generative Pre-Training (GPT-1)** — Radford et al., 2018.
- [ ] 12. **Language Models are Unsupervised Multitask Learners (GPT-2)** — Radford et al., 2019.
- [ ] 13. **Language Models are Few-Shot Learners (GPT-3)** — Brown et al., 2020.
- [ ] 14. **BERT: Pre-training of Deep Bidirectional Transformers** — Devlin et al., 2018.
- [ ] 15. **Exploring the Limits of Transfer Learning (T5)** — Raffel et al., 2019.
- [ ] 16. **LLaMA: Open and Efficient Foundation Language Models** — Touvron et al., 2023.
- [ ] 17. **Llama 2: Open Foundation and Fine-Tuned Chat Models** — Touvron et al., 2023.
- [ ] 18. **Fast Transformer Decoding: One Write-Head is All You Need (MQA)** — Shazeer, 2019.
- [ ] 19. **GQA: Training Generalized Multi-Query Transformer Models** — Ainslie et al., 2023.
- [ ] 20. **Switch Transformer** — Fedus et al., 2021 — sparse MoE at scale.
- [ ] 21. **Mixtral of Experts** — Jiang et al., 2024 — open MoE done right.
- [ ] 22. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Gu & Dao, 2023.

## Pre-Training (9)

- [ ] 23. **Scaling Laws for Neural Language Models** — Kaplan et al., 2020.
- [ ] 24. **Training Compute-Optimal Large Language Models (Chinchilla)** — Hoffmann et al., 2022.
- [ ] 25. **ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models** — Rajbhandari et al., 2019.
- [ ] 26. **Megatron-LM** — Shoeybi et al., 2019 — tensor parallelism.
- [ ] 27. **GPipe** — Huang et al., 2018 — pipeline parallelism.
- [ ] 28. **PaLM: Scaling Language Modeling with Pathways** — Chowdhery et al., 2022.
- [ ] 29. **PyTorch FSDP** — Zhao et al., 2023.
- [ ] 30. **Gopher / Scaling Language Models** — Rae et al., 2021.
- [ ] 31. **Mixed Precision Training** — Micikevicius et al., 2017.

## Post-Training — RL & Preference Learning (10)

- [ ] 32. **Proximal Policy Optimization (PPO)** — Schulman et al., 2017.
- [ ] 33. **Deep RL from Human Preferences** — Christiano et al., 2017.
- [ ] 34. **Learning to Summarize from Human Feedback** — Stiennon et al., 2020.
- [ ] 35. **Training Language Models to Follow Instructions (InstructGPT)** — Ouyang et al., 2022.
- [ ] 36. **Constitutional AI: Harmlessness from AI Feedback** — Bai et al., 2022.
- [ ] 37. **Direct Preference Optimization (DPO)** — Rafailov et al., 2023.
- [ ] 38. **DeepSeekMath / Group Relative Policy Optimization (GRPO)** — Shao et al., 2024.
- [ ] 39. **RLAIF: Scaling RL from AI Feedback** — Lee et al., 2023.
- [ ] 40. **KTO: Kahneman-Tversky Optimization** — Ethayarajh et al., 2024.
- [ ] 41. **A General Theoretical Paradigm to Understand Learning from Preferences (IPO)** — Azar et al., 2023.

## Fine-Tuning (6)

- [ ] 42. **LoRA: Low-Rank Adaptation of LLMs** — Hu et al., 2021.
- [ ] 43. **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al., 2023.
- [ ] 44. **Prefix-Tuning** — Li & Liang, 2021.
- [ ] 45. **The Power of Scale for Parameter-Efficient Prompt Tuning** — Lester et al., 2021.
- [ ] 46. **DoRA: Weight-Decomposed Low-Rank Adaptation** — Liu et al., 2024.
- [ ] 47. **Model Soups** — Wortsman et al., 2022 — weight averaging.

## Reasoning (8)

- [ ] 48. **Chain-of-Thought Prompting Elicits Reasoning** — Wei et al., 2022.
- [ ] 49. **Self-Consistency Improves CoT** — Wang et al., 2022.
- [ ] 50. **Tree of Thoughts** — Yao et al., 2023.
- [ ] 51. **STaR: Self-Taught Reasoner** — Zelikman et al., 2022.
- [ ] 52. **Let's Verify Step by Step (PRMs)** — Lightman et al., 2023.
- [ ] 53. **Math-Shepherd** — Wang et al., 2023 — process rewards without human labels.
- [ ] 54. **Scaling LLM Test-Time Compute Optimally** — Snell et al., 2024.
- [ ] 55. **DeepSeek-R1: Incentivizing Reasoning via RL** — DeepSeek, 2025.

## Systems (Training Infra) (5)

- [ ] 56. **Ray: A Distributed Framework for Emerging AI Applications** — Moritz et al., 2018.
- [ ] 57. **FlashAttention** — Dao et al., 2022 — IO-aware exact attention.
- [ ] 58. **FlashAttention-2** — Dao, 2023.
- [ ] 59. **Alpa: Automating Inter- and Intra-Operator Parallelism** — Zheng et al., 2022.
- [ ] 60. **Using DeepSpeed and Megatron to Train MT-NLG 530B** — Smith et al., 2022.

## Inference (9)

- [ ] 61. **Efficient Memory Management for LLM Serving (vLLM / PagedAttention)** — Kwon et al., 2023.
- [ ] 62. **Fast Inference from Transformers via Speculative Decoding** — Leviathan et al., 2022.
- [ ] 63. **Medusa: Multi-Head Decoding** — Cai et al., 2024.
- [ ] 64. **EAGLE: Speculative Sampling with Feature Uncertainty** — Li et al., 2024.
- [ ] 65. **Orca: Continuous Batching** — Yu et al., 2022.
- [ ] 66. **SGLang: Efficient Execution of Structured Language Programs** — Zheng et al., 2023.
- [ ] 67. **DistServe: Disaggregating Prefill and Decode** — Zhong et al., 2024.
- [ ] 68. **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** — Patel et al., 2023.
- [ ] 69. **Break the Sequential Dependency (Lookahead Decoding)** — Fu et al., 2024.

## Quantization (8)

- [ ] 70. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** — Dettmers et al., 2022.
- [ ] 71. **GPTQ: Accurate Post-Training Quantization** — Frantar et al., 2022.
- [ ] 72. **AWQ: Activation-aware Weight Quantization** — Lin et al., 2023.
- [ ] 73. **SmoothQuant** — Xiao et al., 2022.
- [ ] 74. **FP8 Formats for Deep Learning** — Micikevicius et al., 2022.
- [ ] 75. **Microscaling Data Formats (MXFP)** — OCP, 2023.
- [ ] 76. **BitNet: Scaling 1-bit Transformers** — Wang et al., 2023.
- [ ] 77. **QuIP: 2-Bit Quantization with Incoherence Processing** — Chee et al., 2023.

## Data (5)

- [ ] 78. **The Pile** — Gao et al., 2020.
- [ ] 79. **RedPajama** — Together, 2023 — open pretraining corpus.
- [ ] 80. **FineWeb: Decanting the Web for the Finest Text Data** — Penedo et al., 2024.
- [ ] 81. **Textbooks Are All You Need (phi-1)** — Gunasekar et al., 2023.
- [ ] 82. **Dolma: An Open Corpus of Three Trillion Tokens** — Soldaini et al., 2024.

## Multimodal (7)

- [ ] 83. **CLIP: Learning Transferable Visual Models from Natural Language Supervision** — Radford et al., 2021.
- [ ] 84. **Flamingo: a Visual Language Model for Few-Shot Learning** — Alayrac et al., 2022.
- [ ] 85. **BLIP-2** — Li et al., 2023.
- [ ] 86. **LLaVA: Visual Instruction Tuning** — Liu et al., 2023.
- [ ] 87. **PaLI: Scaling Language-Image Learning** — Chen et al., 2022.
- [ ] 88. **Qwen-VL** — Bai et al., 2023.
- [ ] 89. **Sora / Video Diffusion Models overview** — OpenAI, 2024 (if you want generative video context).

## Agents & Tool Use (4)

- [ ] 90. **ReAct: Synergizing Reasoning and Acting** — Yao et al., 2022.
- [ ] 91. **Toolformer: Language Models Can Teach Themselves to Use Tools** — Schick et al., 2023.
- [ ] 92. **Voyager: An Open-Ended Embodied Agent with LLMs** — Wang et al., 2023.
- [ ] 93. **SWE-agent / SWE-bench** — Yang et al., 2024.

## Evaluation (3)

- [ ] 94. **MMLU: Measuring Massive Multitask Language Understanding** — Hendrycks et al., 2020.
- [ ] 95. **HELM: Holistic Evaluation of Language Models** — Liang et al., 2022.
- [ ] 96. **Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference** — Chiang et al., 2024.

## Safety & Alignment (5)

- [ ] 97. **Red Teaming Language Models with Language Models** — Perez et al., 2022.
- [ ] 98. **Universal and Transferable Adversarial Attacks on Aligned LMs (GCG)** — Zou et al., 2023.
- [ ] 99. **Sleeper Agents: Training Deceptive LLMs that Persist** — Hubinger et al., 2024.
- [ ] 100. **Weak-to-Strong Generalization** — Burns et al., 2023.

## Interpretability (bonus, not counted — highly recommended)

- [ ] *A Mathematical Framework for Transformer Circuits* — Elhage et al., 2021.
- [ ] *In-context Learning and Induction Heads* — Olsson et al., 2022.
- [ ] *Toy Models of Superposition* — Elhage et al., 2022.
- [ ] *Towards Monosemanticity (SAEs)* — Bricken et al., 2023.
- [ ] *Scaling Monosemanticity (Claude 3 Sonnet SAEs)* — Templeton et al., 2024.
- [ ] *Interpreting GPT: The Logit Lens* — nostalgebraist, 2020 (blog post).

---

## How to Use This List

1. Pick the section you want to go deep on.
2. Read top-down — earlier papers set up the machinery that later ones assume.
3. After each paper, extract the concepts into the relevant topic folder following [`TEMPLATE.md`](TEMPLATE.md).
4. If a paper is a milestone or end-to-end system, add a writeup in [`case-studies/`](case-studies/) that *links to* the concept pages rather than duplicating them.
5. Check the box when the paper's concepts are in the repo.
