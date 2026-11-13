# Self-learning RL (reference-navigation agents)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An RL recipe that teaches a model to *autonomously read* a long reference (a textbook, a documentation set) while solving a problem, by rewarding fine-grained navigation and reading decisions, not just the final answer. *Incentivizing Self-Learning in LLMs via Reinforcement Learning* (2610.39671) lifts Intern-S2-Preview-35B from 51.5% → 79.5% on math proofs and transfers to out-of-domain tasks (+1.7 on BrowseComp-Plus, +2.3 on CL-bench).

**Prereqs:** [../rlvr.md](../rlvr.md), [prm.md](prm.md)
**Related:** [long-cot-rl.md](long-cot-rl.md), [../grpo.md](../grpo.md), [../../agents/README.md](../../agents/README.md)

---

## What it is

A long-form reference is available but no reference solution is. The model must decide: which chapter to look at, which section to search inside, how much to read, when to stop and answer. The agent has only a small, well-typed tool interface over the reference (TOC navigation, section search, read-section), and a verifiable task (the proof is either correct or not).

Self-learning RL is the training recipe that makes those decisions improve under RL.

## How it works

1. **Minimal tool interface.** `list_toc(book)`, `search(query)`, `read(section_id)`, `answer(...)`. No free-text search over the whole book, no scratchpad outside the trajectory.
2. **Fine-grained rewards.** Process rewards score each navigation step: did this search surface a section that was actually read? did this read contain the lemma used in the final proof? Final answer still gets the largest reward.
3. **Policy update.** On-policy RL (GRPO-style group baselines are a natural fit) over trajectories that span many tool calls. The long rollouts make partial rewards essential — pure outcome rewards would be too sparse.
4. **Negative data from self-play.** Failed trajectories feed back as negatives after being annotated by the process-reward model.

The combination gets the agent to learn a *reading strategy* rather than memorizing a lucky chain.

## Why it matters

- Pairs naturally with [textbook-structured SFT](../../data/textbook-structured-sft.md): one builds a well-typed reference corpus, the other learns to navigate it at inference time.
- Suggests "retrieval + reasoning" works better as a *policy trained end-to-end over tool trajectories* than as a two-stage retrieve-then-reason pipeline. Transfer to OOD search tasks supports that.
- Offers a cleaner alternative to RAG fine-tuning: the model owns the retrieval decisions and can trade off reading breadth vs. depth.

## Gotchas & tricks

- Process-reward model has to be more accurate than the policy on step quality, or the training signal is noise.
- Keep the tool interface small and discrete — adding free-text search explodes the policy's action space and destabilizes RL.
- Reward shaping that credits "section ended up read" too generously causes aimless browsing; attach the credit to *use* in the final answer.
- Transfer only shows up when the training reference is genuinely structured — flat corpora do not teach navigation.

## Sources

- Paper: *Incentivizing Self-Learning in Large Language Models via Reinforcement Learning* — Gu, Zhao, Wu, Fang, Hua, Sun, Chen, Zhang (2026) — [arXiv:2610.39671](https://arxiv.org/abs/2610.39671)
