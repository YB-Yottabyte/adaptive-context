# Context Debug

Context Debug is a CSE 598 Agentic AI capstone project about context engineering
for repository-level software debugging. This first baseline retrieves repository
context once and asks one configured coding model to diagnose the bug and propose a
fix. Groq with GPT-OSS 20B is the primary runnable baseline. It never edits
repository files.

The retrieval design adapts one idea from *RepoCoder: Repository-Level Code
Completion Through Iterative Retrieval and Generation* (Zhang et al., EMNLP 2023):
split repository code into snippets, retain source-file metadata, and retrieve the
most relevant snippets before generation. This baseline deliberately uses only the
initial retrieval-augmented generation step, not RepoCoder's iterative process.

## Baseline architecture

```text
Bug report
    -> Python repository chunks
    -> one-shot TF-IDF + cosine-similarity ranking
    -> Top-K chunks
    -> debugging prompt
    -> one configured model provider
    -> root cause + proposed fix
```

Code is split into deterministic, overlapping line chunks. Retrieval uses a small
standard-library TF-IDF implementation, so no embedding API, vector database, or
external retrieval service is needed. Ties are resolved by source path and line
number for reproducible results.

The primary model is Groq's `openai/gpt-oss-20b`. An optional local MLX provider is
available for Apple Silicon experiments, but it is not required to install, run, or
grade the baseline.

## Requirements and setup

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A Groq API key for the canonical baseline run

Install the locked runtime and development dependencies:

```bash
uv sync --dev
```

The standard install includes the Groq SDK, Rich, and Questionary. On Apple Silicon,
it also installs the MLX runtime so the local provider is immediately selectable.
The platform marker keeps MLX out of installations on unsupported systems, and the
canonical Groq baseline does not depend on local inference.

## Run a debugging test case

The mini-repository in `test_cases/payment_bug/` intentionally contains a bug.
Confirm its behavioral test fails with:

```bash
uv run pytest test_cases/payment_bug
```

Do not fix the test case: its failing test is the input to this debugging study.
The root-level `tests/` directory checks whether the baseline implementation works;
`test_cases/` evaluates whether the baseline can identify bugs in other projects.

Configure the primary baseline in `.env` without committing the API key:

```dotenv
GROQ_API_KEY=your-secret-key
GROQ_MODEL=openai/gpt-oss-20b
```

Export those values into the current shell:

```bash
set -a
source .env
set +a
```

For a normal class or demo run, pass the test-case name directly:

```bash
uv run python src/baseline.py payment_bug
```

The short name resolves to `test_cases/payment_bug`. The CLI asks for Top-K and,
when multiple providers are configured, asks which provider to use. Explicit paths
such as `test_cases/payment_bug` are also accepted.

For a non-interactive or reproducible run, specify ambiguous choices explicitly:

```bash
uv run python src/baseline.py payment_bug \
    --provider groq \
    --model openai/gpt-oss-20b \
    --top-k 3
```

## Provider selection

The CLI detects configured providers before running. Groq is available when
`GROQ_API_KEY` is set and the Groq SDK is installed. The local provider is available
on Apple Silicon when MLX is installed; it uses `CONTEXT_DEBUG_MODEL`, `--model`, or
the repository's default local model in that order.

With only Groq configured, omitting `--provider` automatically selects it without a
menu. The normal startup then identifies the selected model as
`Using Groq — GPT-OSS 20B`.

```bash
uv run python src/baseline.py payment_bug
```

When both providers are available and `--provider` is omitted in a terminal, an
arrow-key selector is shown. In non-interactive environments with multiple
providers, pass `--provider` explicitly. Top-K is likewise requested interactively
when omitted; use `--top-k` in scripts or to compare retrieval sizes:

```bash
uv run python src/baseline.py payment_bug --top-k 5
```

### Local provider

On an Apple Silicon Mac, `uv sync` installs MLX. Optionally override the default
Hugging Face model:

```bash
export CONTEXT_DEBUG_MODEL=mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
```

Run local inference explicitly:

```bash
uv run python src/baseline.py payment_bug --provider local
```

To exercise the interactive selector, configure Groq as above, configure the local
model, and omit `--provider`:

```bash
uv run python src/baseline.py payment_bug
```

On its first run, `mlx-lm` may download the configured model from Hugging Face. The
local and Groq providers receive the same retrieved Top-K context and debugging
prompt, and each run still performs exactly one model request or generation.

The input directory must contain a non-empty `bug_report.txt` and at least one
Python file. The terminal uses transient status lines while it searches, prepares
the selected context, and waits for the model. Those status lines disappear when
each stage finishes. As normal response chunks arrive, one Rich live view updates
in place and finishes as a concise public evidence summary. It does not expose or
request hidden model reasoning. The suggested code contains the complete rewritten
function, or the smallest self-contained block when the change is not inside a
function. The generated fix is advisory only; no files are modified.

```text
Context Debugging Baseline

Using Groq — GPT-OSS 20B

I'll trace the reported behavior through the most relevant files and look for
the smallest fix.

● Retrieved 3 relevant files
  ├─ test_payment.py
  ├─ payment.py
  └─ discount.py

● Diagnosis points to payment.py

  Evidence
  ├─ test_payment.py defines the expected discounted result.
  ├─ payment.py contains checkout().
  └─ discount.py correctly computes the discounted value.

  checkout() discards the value returned by apply_discount(), so it returns
  the original price unchanged.

  Suggested change

  5 │     apply_discount(price, discount)
  6 │     return price
  5 └────→     return apply_discount(price, discount)

  apply_discount() already calculates the correct price. Returning that value
  fixes checkout().

  345 input · 243 output · 588 total tokens
  Completed in 1.8s
```

The provider's real stream is measured first, then the completed response is
revealed at a steady reading pace. Presentation playback does not inflate model
latency measurements. The result shows a concise line-level change when the
diagnosed definition is present in the selected repository context. Total execution
time covers the actual end-to-end baseline run, including terminal presentation.

Run the non-LLM unit tests and lint checks with:

```bash
uv run pytest
uv run ruff check .
```

## Current limitations

- Retrieval happens exactly once from the original natural-language bug report.
- Chunking is based on fixed line windows rather than Python syntax.
- Sparse lexical retrieval may miss conceptually related code with different names.
- The prompt has no explicit token-budget enforcement beyond Top-K selection.
- The model proposes a fix but does not run tests or modify code.
- Local generation speed and quality depend on the selected model and hardware.

## Future direction

The capstone will explore dynamic context selection informed by test failures,
stack traces, and previous findings. Planned extensions include context pruning,
context summarization, iterative context updates, and token-budget optimization.
Those features are intentionally outside this minimal baseline.

## Generative AI use

Codex was used to assist with implementation. Generated code was reviewed and
tested by the author.
