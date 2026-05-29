# Sources — HF Daily Papers + arXiv

Reference for the URLs to hit, what they return, and how to parse them with `WebFetch`.

## Hugging Face Daily Papers (primary)

HF's `papers/date/<YYYY-MM-DD>` page lists papers that have been submitted to HF (typically by authors) on that date, ordered by community upvotes. Strong relevance signal because it's curated by an LLM-focused audience.

### URLs

- **Daily list:** `https://huggingface.co/papers/date/YYYY-MM-DD`
  - Example: `https://huggingface.co/papers/date/2026-05-29`
  - Returns: a list of papers, each linking to `/papers/<arxiv-id>`.
  - Empty on weekends / holidays.
- **Per-paper page:** `https://huggingface.co/papers/<arxiv-id>`
  - Example: `https://huggingface.co/papers/2401.04088`
  - Returns: title, authors, abstract, comments, related artifacts, and crucially an `og:image` meta tag pointing to the hero figure.

### Parsing tips for `WebFetch`

`WebFetch` returns HTML rendered to markdown-ish text. Prompts that work well:

For the daily list:
> "List all papers on this page. For each, output one line: `<arxiv-id> | <title> | <upvotes if present>`. The arXiv ID is the last path segment of each `/papers/...` link."

For the per-paper page:
> "Extract: (1) the paper title, (2) the full author list, (3) the institutional affiliation(s) if shown, (4) the abstract — exact text, no summary, (5) the URL in the `og:image` meta tag, (6) the URL in the `og:url` tag."

If the `og:image` extraction fails, look for the first large image on the page — HF typically renders a figure preview near the top.

## arXiv (fallback + completeness check)

Use when HF returns 0 or <5 papers, or when a key topic appears underrepresented.

### URLs

- **Daily new-submissions** by category:
  - `https://arxiv.org/list/cs.LG/YYYY-MM-DD` — machine learning
  - `https://arxiv.org/list/cs.CL/YYYY-MM-DD` — computation and language
  - `https://arxiv.org/list/cs.AI/YYYY-MM-DD` — artificial intelligence
  - Older format `cs.LG/YYYY-MM` (monthly) also works for browsing — but prefer daily.
- **Per-paper abstract:** `https://arxiv.org/abs/<id>`
  - Example: `https://arxiv.org/abs/2305.12345`
- **PDF:** `https://arxiv.org/pdf/<id>` (don't fetch for the digest — abstract is enough).

### Parsing tips

For a daily category page:
> "List every paper entry. For each, output one line: `<arxiv-id> | <title> | <one-line subject classification>`. arXiv IDs look like `2305.12345` or `2305.12345v1` — strip the version suffix if present."

For an abstract page:
> "Extract: (1) title, (2) full author list, (3) primary subject classification, (4) abstract verbatim, (5) submission date."

arXiv pages don't reliably expose a hero figure URL — you typically can't get a figure without downloading the PDF and extracting. If HF doesn't have the paper either, **skip the figure** rather than parsing a PDF.

### Daily-list URL quirk

arXiv's `/list/<cat>/YYYY-MM-DD` returns submissions whose listing date matches that day — usually one US business day's worth. If you get nothing, try the surrounding two days (timezone slop, late announcements).

## Fallback decision tree

```
HF page exists & lists ≥ 5 papers
  → primary source = HF
  → header: "Curated from HF Daily Papers"

HF page exists & lists 1–4 papers
  → primary = HF, fallback = arXiv cs.LG/CL/AI for the same date
  → header: "Curated from HF Daily Papers + arXiv (cs.LG/CL/AI)"

HF page 404s or lists 0 papers
  → arXiv only, same date
  → header: "Curated from arXiv (cs.LG/CL/AI) — HF Daily Papers empty for this date"

arXiv also returns nothing
  → still write the digest, but with a single section "No papers found for this date"
  → don't fabricate
```

## Hero figure download

Once you have the figure URL (HF `og:image` is the only reliable one):

```bash
mkdir -p daily-papers/assets/<YYYY-MM-DD>
curl -sL --max-time 15 "<image-url>" -o "daily-papers/assets/<YYYY-MM-DD>/<arxiv-id>.<ext>"
```

- Extension: parse from URL (`.png`, `.jpg`, `.jpeg`, `.webp`). Default to `.png`.
- Validate: after `curl`, check the file exists and is `> 1000` bytes (`stat -f%z "<path>"` on macOS). If not, delete it and omit the `![Hero figure]` line for that paper.
- Some HF og:images redirect — `curl -sL` follows redirects, which is what you want.

## Common pitfalls

- **arXiv IDs with version suffixes:** `2305.12345v2` — strip `v\d+` for filenames and link consistency.
- **Cross-listed papers:** the same paper can appear under `cs.LG` and `cs.CL`. Dedup by arXiv ID before per-paper fetches.
- **Withdrawn papers:** arXiv keeps the listing but the abstract page says "withdrawn." Skip these.
- **Rate limiting:** if a burst of `WebFetch` to HF starts failing, slow down and retry — don't loop tightly.
- **HF og:image absence:** some papers have no preview image set. Just skip the figure for that one paper.
