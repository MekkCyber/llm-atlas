---
name: daily-papers
description: Fetch the most important AI papers (LLMs, GenAI, diffusion, RL, safety, interp, agents, multimodal) for a given date, summarize each, and map them to the llm-atlas knowledge graph. Asks for the date if not provided.
allowed-tools: Read, Glob, Grep, Bash, WebFetch, Write
---
# Daily AI papers digest

You are producing a hand-curated daily digest of AI papers for the `llm-atlas` knowledge graph (a "how models are trained" wiki). The output is one markdown file per date at `daily-papers/<YYYY-MM-DD>.md` in the repo, with each paper summarized and explicitly tied back to existing concept pages in the graph.

The graph is hand-maintained — **never edit existing concept files, `READING-LIST.md`, or `PAPERS.md`**. The digest only *suggests* new pages and links existing ones.

---

## Workflow

### 1. Resolve target date

- If the user passed a date in arguments (`/daily-papers 2026-05-29`, `/daily-papers yesterday`, `/daily-papers today`), parse it to `YYYY-MM-DD`. Use `date` in Bash for relative forms (`date -v-1d +%Y-%m-%d` for yesterday on macOS).
- If no argument: call `AskUserQuestion` with options **today**, **yesterday**, **custom date**. If the user picks custom, ask for the date in `YYYY-MM-DD` form.
- Reject dates in the future — papers don't exist yet. Tell the user and re-ask.

### 2. Locate the llm-atlas repo

- The repo root contains `PAPERS.md`, `READING-LIST.md`, and a `case-studies/` directory. Check the current working directory first.
- If not in the repo, ask the user for the repo path. Default suggestion: `/Users/mekkcyber/Dreams/knowledge/llm-atlas`.
- All paths below are relative to the repo root. `cd` there if needed.

### 3. Fetch HF Daily Papers (primary source)

Use `WebFetch` on `https://huggingface.co/papers/date/<YYYY-MM-DD>` with a prompt like:

> "Extract all paper entries on this page. For each, return: title, arXiv ID (from the `/papers/<id>` link), and the upvote count if visible. Return as a list, one paper per line: `<arxiv-id> | <title> | <upvotes>`."

For each arXiv ID returned, `WebFetch` `https://huggingface.co/papers/<arxiv-id>` with:

> "Extract: the paper title, the full author list with affiliations if shown, the abstract verbatim, and the URL of the main figure (look for an `og:image` meta tag or a hero image at the top of the page)."

If the date page returns **0 papers** (weekends, holidays) or **<5 papers**, fall back to arXiv (next step) — but always still try HF first, since HF curates by community upvotes which is a strong signal.

### 4. arXiv fallback

When HF coverage is thin, query the daily new-submissions page directly:

- `https://arxiv.org/list/cs.LG/<YYYY-MM-DD>` (machine learning)
- `https://arxiv.org/list/cs.CL/<YYYY-MM-DD>` (computation and language)
- `https://arxiv.org/list/cs.AI/<YYYY-MM-DD>` (AI)

For each `WebFetch`, prompt:

> "Extract every paper entry on this page. Return one per line as `<arxiv-id> | <title> | <one-line subject area>`."

Then `WebFetch https://arxiv.org/abs/<id>` per candidate for the full abstract and authors. See `references/sources.md` for the URL shapes and parsing quirks.

If the digest header ends up arXiv-only, say so explicitly in the "Curated from" line.

### 5. Filter for relevance

Apply the keep/skip rules in `references/relevance-filter.md`. The high-level filter:

- **Keep:** LLMs, multimodal LLMs, diffusion (text/image/video), RL post-training, reasoning, agents/tool use, safety/alignment, interpretability, efficient inference, quantization, evaluation/benchmarks for the above.
- **Skip:** pure computer vision (non-generative), classical NLP without LLM angle, robotics-only (unless agent/foundation-model-flavored), theory disconnected from neural nets, application papers that don't introduce a new technique.

When in doubt, **keep**, and note the borderline call in the section. The user wants a wide net filtered for *topical fit*, not a tight cap.

### 6. Build the knowledge-graph index (once, reuse for all papers)

Before producing per-paper sections, build a search-friendly index of the existing graph:

```bash
# from repo root
```

Use `Glob '**/*.md'` excluding `node_modules/`, `nimbalyst-local/`, `daily-papers/`, and any other non-content dirs. Read `references/kg-structure.md` for the layout (`fundamentals/`, `architectures/`, `pre-training/`, `post-training/{,fine-tuning,reasoning}/`, `systems/`, `inference/`, `quantization/`, `data/`, `multimodal/`, `agents/`, `evaluation/`, `safety/`, `interpretability/`, `case-studies/`).

For each kept paper, run a small batch of `Grep` queries against the repo for **distinctive terms** drawn from the title + abstract: technique names ("GRPO", "MLA", "DualPipe"), model families ("Llama", "DeepSeek", "Qwen"), author orgs ("Anthropic", "DeepMind"), method classes ("rejection sampling", "speculative decoding"). Combine the file matches into the candidate "Related existing pages" set.

**Do not re-glob the repo per paper.** One `Glob` up front; many `Grep` calls is fine.

### 7. Per-paper section

Use `templates/digest-template.md` as the shape for each section. Fill in:

- **Title** (verbatim from HF/arXiv).
- **Authors + organization** — prefer the org from author affiliations; if multiple, list the primary one.
- **Links** — arXiv, HF (omit if arXiv-only), and AlphaXiv (`https://www.alphaxiv.org/abs/<arxiv-id>`).
- **Topics** — short backtick tags, drawn from the keep-list (`LLM`, `RL`, `safety`, `interp`, `MoE`, `diffusion`, `agents`, `multimodal`, `quantization`, `efficient-inference`, `reasoning`, `eval`).
- **Hero figure** — see below.
- **TL;DR** — one paragraph, in your own words. Don't copy the abstract.
- **Key idea** — the core mechanism / insight. Be concrete — name the technique, the loss, the trick.
- **Main finding** — what they measured, what improved, by how much. Numbers preferred.
- **Why it matters** — field-level implication. What does this unblock, what does it challenge.
- **Knowledge graph integration:**
  - **Builds on / relates to existing pages:** real markdown links to files you found via Grep. Each link gets a one-line "why this is related" (e.g. *extends GRPO with a learned value baseline*). Use the path **relative to `daily-papers/`** — i.e. `../post-training/grpo.md`.
  - **Suggested new pages:** propose `<folder>/<filename>.md *(depth)*` or `<folder>/<filename>.md *(taxonomy)*` or `case-studies/<filename>.md *(case study)*` entries. Follow llm-atlas naming: depth files are kebab-case technique names, taxonomy files start with `_`, case studies are kebab-case model names. **Do not create these files.** Just list them.

Skip the "Suggested new pages" subsection if the paper is purely incremental and adds nothing the graph doesn't already cover.

### 8. Download the hero figure

For each paper with an `og:image` URL from HF (or an arXiv figure URL):

```bash
mkdir -p daily-papers/assets/<YYYY-MM-DD>
curl -sL --max-time 15 "<image-url>" -o "daily-papers/assets/<YYYY-MM-DD>/<arxiv-id>.<ext>"
```

- Infer extension from the URL (`.png`, `.jpg`, `.webp`); default to `.png` if ambiguous.
- If `curl` fails or the file is empty / <1KB, drop the `![Hero figure]` line for that paper (don't leave a broken reference). Note the missing figure in the final report.
- Use `arxiv-id` without dots replaced — keep the original `2305.12345` form. Filenames are safe.

### 9. Write the digest

`Write` to `daily-papers/<YYYY-MM-DD>.md` using `templates/digest-template.md`. Fill the header:

- Date.
- Sources used (`HF Daily Papers`, `arXiv cs.LG/CL/AI`, or both).
- `M of N` count: M kept, N total fetched before filtering.

Then the "Papers at a glance" table — one row per kept paper, with title linked to the in-page anchor (`#<n>-<slug>`), topic tags, and org.

Then each numbered paper section in order.

**If the file already exists for this date:** ask the user via `AskUserQuestion` whether to **overwrite** or **abort**. Default to overwrite if the user already explicitly invoked with a date.

### 10. Final report

Print to the user:

- Absolute path to the new digest file.
- Counts: **N** total fetched, **M** kept after relevance filter, **K** with hero figures successfully downloaded.
- A bullet list of **suggested new depth / case-study files** aggregated across all papers (so the user can pick what to draft next).
- One-line note on `git status` impact: only new files under `daily-papers/` should appear; no edits to existing concept files, `READING-LIST.md`, or `PAPERS.md`.

---

## Hard rules

- **Never write to or edit** any existing concept file, `READING-LIST.md`, `PAPERS.md`, or topical `README.md`. Only suggest.
- **Never invent papers.** If a date has no papers, produce a digest with a clear "no papers found" note rather than fabricating entries.
- **Never invent KG links.** Every `../<folder>/<file>.md` link in the digest must be a real file you found via Glob/Grep. Verify with `Read` or `ls` if uncertain.
- **One Glob, many Greps.** Don't re-glob the repo per paper — it's wasteful and slow.
- **Quote conservatively.** Summarize abstracts in your own words; don't paste large chunks. Verbatim title is fine.

## Reference files

- [kg-structure.md](./references/kg-structure.md) — llm-atlas conventions: depth vs taxonomy, folder map, naming rules.
- [sources.md](./references/sources.md) — HF / arXiv URL shapes, extraction tips, fallback logic.
- [relevance-filter.md](./references/relevance-filter.md) — keep/skip table with example keywords.
- [digest-template.md](./templates/digest-template.md) — skeleton for the daily output file.
