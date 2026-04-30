# BFCL — Berkeley Function-Call Leaderboard
*Depth — evaluate tool-use / function-calling capability.*

**TL;DR:** Evaluates a model's ability to **generate correctly-formed function calls** given a tool schema and a user query. Multiple categories: simple calls, parallel calls, multiple-tool calls, nested calls, multi-turn tool use. AST-based grading checks call signatures, argument values, types. Maintained by UC Berkeley's Gorilla team. The standard function-calling benchmark for frontier models; Llama 3.1 405B scores **88.5%**; Claude 3.5 Sonnet: 90.2% (leading). Not saturated.

**Prereqs:** *(none)*
**Related:** [mmlu-pro](mmlu-pro.md) · [humaneval](humaneval.md)

---

## What it is

Yan et al., the Berkeley Function-Calling Leaderboard, first described in a blog post (2024) and formalized in subsequent releases. Site: https://gorilla.cs.berkeley.edu/leaderboard.html. Repo: https://github.com/ShishirPatil/gorilla.

Evaluates "tool use" / "function calling": given a prompt like *"What's the weather in Paris?"* and a set of tools (e.g., `get_weather(city: str, units: str)`), does the model emit a correctly-formatted function call?

### Categories

BFCL v3 (latest as of 2025) has multiple categories:

- **Simple**: one function available, one call needed.
- **Multiple**: several functions available, pick the right one.
- **Parallel**: multiple calls in one response (e.g., weather in Paris AND London).
- **Parallel-multiple**: multiple different tools called in parallel.
- **Irrelevance**: query doesn't need a tool — model should NOT call one.
- **Live**: real-world tool-use traces from production systems.
- **Multi-turn**: back-and-forth tool use with prior tool results.
- **Executable**: calls are actually executed; check the *output* matches, not just the call.

### Grading

AST-based: parse the generated function call, extract the function name and arguments, check:
- Function name matches ground-truth.
- Required arguments present.
- Argument values match (with type coercion).
- Parallel calls match as an unordered set.

Pass = all checks pass. Score = fraction of test cases passing.

For Executable category: actually execute the call (with a sandbox), check result.

---

## How it works as an LLM eval

### Format

- Input: user query + list of available tools (JSON schema or Python function signatures).
- Output: tool call (structured — either JSON, or Python function call syntax, or the model's native tool-call format).
- Grading: AST comparison to ground truth.

### Prompts

The original BFCL uses both:
- **Prompt-based**: tool definitions embedded in the system prompt; model emits calls in a specific string format.
- **Native function calling**: using the model's API-level tool-calling format (e.g., OpenAI's `tools=[...]` parameter, Anthropic's tool-use format). Treats function calling as a first-class capability of the API.

BFCL grades both patterns.

### Common scoring convention

**Overall accuracy** = macro-average across categories.

---

## Why it matters

- **The standard tool-use benchmark.** Most post-2023 tech reports quote BFCL scores (Llama 3, DeepSeek V3, Kimi k1.5, Claude, etc.).
- **Not saturated at the frontier.** Top scores ~90%; headroom remains for multi-turn and executable categories.
- **Measures a production-critical capability.** Agents, autonomous systems, RAG-with-tools — all depend on reliable function calling.
- **Category breakdown is informative.** A model strong on "Simple" but weak on "Parallel" or "Multi-turn" tells you something specific about its capability profile.

---

## Gotchas & tricks

- **Prompt format matters.** Models trained with one tool-call format (OpenAI's vs Anthropic's vs XML) perform differently on BFCL. Report the format used.
- **Multi-turn is much harder than single-turn.** Scores drop 10–20 pp from single-turn to multi-turn categories. Often under-reported.
- **Executable requires a sandbox.** Some orgs don't run Executable; they just grade at AST level.
- **Argument type coercion is brittle.** A model outputting `"42"` vs `42` vs `"42.0"` for an int parameter may fail grading even though the semantic intent is correct. BFCL's grader handles the common cases but not all.
- **Contamination risk.** BFCL's test set is public; models can be fine-tuned on it. The Live category (continuously refreshed with production traces) is the contamination-resistant version.
- **Not a distributional eval.** BFCL tests correctness of calls, not whether the model chooses the *right* tool in ambiguous situations.

---

## Typical modern numbers

| Model | BFCL Overall |
|---|---|
| Claude 3.5 Sonnet | 90.2% |
| GPT-4 (0125) | 88.3% |
| Llama 3.1 405B | 88.5% |
| Nemotron 4 340B | 86.5% |
| Llama 3.1 70B | 84.8% |
| GPT-4o | 80.5% |
| Qwen 2.5 72B | — |
| Kimi k1.5 | — (not reported) |
| Llama 3.1 8B | 76.1% |

---

## Sources

- Site: Berkeley Function-Calling Leaderboard — https://gorilla.cs.berkeley.edu/leaderboard.html.
- Repo: Gorilla — https://github.com/ShishirPatil/gorilla.
- Blog: *Berkeley Function Calling Leaderboard* — Yan, Patil et al., 2024.
- Paper: *The Llama 3 Herd of Models* — 2024 — reports BFCL across all three sizes.
