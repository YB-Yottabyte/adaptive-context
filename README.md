<h1 align="center">Adaptive Context</h1>

<p align="center">
  <strong>An AI debugging agent that finds the right code context without overwhelming the model.</strong>
</p>

---

<p align="center">
  <img src="./adaptive-context.png" alt="Adaptive Context diagnosing a payment bug in the terminal" width="100%">
</p>

Adaptive Context solves a specific problem: **helping a developer locate and
explain a behavioral bug without sending an entire repository to an LLM**. It is
designed for developers who can describe or reproduce a failure but do not yet
know which code is responsible.

### Core problem: context compression and sequencing

The core problem is not simply retrieving more code. It is **compressing context
without losing the facts needed to explain the bug** and **sequencing the retained
evidence so the model can follow the failure from the test to the responsible
implementation and its dependencies**.

A successful system produces the correct diagnosis with a smaller, coherently
ordered prompt. It fails when compression removes necessary evidence or poor
sequencing breaks the causal relationships between the failure, execution path,
and root cause.

- **Input:** a bug report plus the repository's source code and tests.
- **Output:** relevant code evidence, the likely root cause and file, and a concise
  suggested change.
- **Success:** the retrieved context contains the code needed to explain the bug,
  the diagnosis identifies the intended faulty behavior, and the suggestion is
  consistent with the failing test.
- **Failure:** the system retrieves insufficient context, identifies the wrong
  cause or file, invents unsupported details, or suggests a change that would not
  satisfy the test.

The current **Static RAG baseline** uses ChromaDB to retrieve the most relevant
code, asks an LLM to diagnose the bug, and returns an evidence-backed suggested
change.

---

## Quickstart

### 1. Install the project

Requirements:

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Internet access for the first embedding-model download and Groq inference
- A Groq API key for the primary baseline

Clone and install the locked dependencies:

```bash
git clone https://github.com/YB-Yottabyte/cse598-capstone-project-proposal.git
cd cse598-capstone-project-proposal
uv sync --dev
```

### 2. Configure a model provider

Groq is the recommended path for reproducing the baseline. Local Hugging Face
inference is optional and requires an Apple Silicon Mac.

#### Option A — Groq and GPT-OSS 20B (recommended)

1. Open [Groq](https://groq.com/) and select **Start Building**, or go directly to
   the [GroqCloud Console](https://console.groq.com/).
2. Create an account or sign in when prompted.
3. In the console's project selector, create a project such as
   `context-debug`, or select an existing project. Groq API keys belong to the
   currently selected project.
4. Open [GroqCloud API Keys](https://console.groq.com/keys).
5. Select **Create API Key**, enter a descriptive name such as
   `context-debug-local`, and create the key.
6. Copy the key and store it securely. Do not paste it into Python source,
   screenshots, issues, or commits.
7. Confirm that **OpenAI GPT-OSS 20B** is available on Groq's
   [supported-models page](https://console.groq.com/docs/models). Its exact API
   model ID is:

   ```text
   openai/gpt-oss-20b
   ```

8. Optionally open the
   [Groq Playground with GPT-OSS 20B selected](https://console.groq.com/playground?model=openai%2Fgpt-oss-20b)
   to try the model in the browser. Selecting a model in the Playground does not
   configure this project; the CLI uses the model ID in `.env` or `--model`.
9. In the repository root, create a local `.env` file containing:

   ```dotenv
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   ```

10. Load those variables into the current terminal session:

    ```bash
    set -a
    source .env
    set +a
    ```

11. Run the reproducible Groq baseline:

    ```bash
    uv run python src/baseline.py payment_bug \
        --provider groq \
        --model openai/gpt-oss-20b \
        --top-k 3
    ```

The project already installs the Groq Python SDK through `uv sync`; no separate
`pip install groq` command is needed. See the official
[Groq API quickstart](https://console.groq.com/docs/quickstart) and
[GPT-OSS 20B model page](https://console.groq.com/docs/model/openai/gpt-oss-20b)
for current provider details.

If the command reports that Groq is not configured, confirm that the current shell
contains the variable without printing the secret itself:

```bash
test -n "$GROQ_API_KEY" && echo "GROQ_API_KEY is set"
```

The `.env` file is ignored by Git. Never commit an API key. If a key is accidentally
published, revoke it in the Groq console and create a replacement.

#### Option B — Local Hugging Face model with MLX (optional)

The local provider runs through Apple's MLX framework and is available only on an
Apple Silicon Mac. Linux, Windows, and Intel Macs should use Groq for this project.

The default local model is the public
[`mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit)
repository. A Hugging Face account or token is normally not required for this
public model.

1. Confirm that the Mac uses Apple Silicon:

   ```bash
   uname -m
   ```

   The result should be `arm64`.

2. Install the project. The platform-specific dependency automatically includes
   `mlx-lm` on supported Macs:

   ```bash
   uv sync --dev
   ```

3. Use the default local model by adding this optional setting to `.env`:

   ```dotenv
   CONTEXT_DEBUG_MODEL=mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
   ```

   Load `.env` again after changing it:

   ```bash
   set -a
   source .env
   set +a
   ```

4. Run the local provider explicitly:

   ```bash
   uv run python src/baseline.py payment_bug \
       --provider local \
       --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
       --top-k 3
   ```

5. On the first run, MLX downloads the model files from Hugging Face and caches
   them locally. The CLI suppresses Hugging Face's nested progress bars so they do
   not disrupt its terminal display; the existing status indicator remains visible
   while loading. Later runs reuse the cache.

To choose another local model, browse the
[MLX Community models on Hugging Face](https://huggingface.co/mlx-community),
choose an MLX-compatible text-generation model that fits the Mac's available
memory, and copy its exact repository ID. Then either pass it once:

```bash
uv run python src/baseline.py payment_bug \
    --provider local \
    --model OWNER/MODEL_REPOSITORY \
    --top-k 3
```

or make it the local default in `.env`:

```dotenv
CONTEXT_DEBUG_MODEL=OWNER/MODEL_REPOSITORY
```

Authentication is needed only when the selected Hugging Face repository is private
or gated:

1. Create or sign in to a [Hugging Face account](https://huggingface.co/join).
2. For a gated model, open its model page while signed in, review its terms, and
   request access. Approval may be immediate or may require the model owner's
   review.
3. Open [Hugging Face Access Tokens](https://huggingface.co/settings/tokens) and
   create a personal token with read access. Write access is unnecessary for model
   downloads.
4. Authenticate the project environment:

   ```bash
   uv run hf auth login
   ```

   Follow the browser/device-code prompt or paste the read token when requested.
5. Verify the signed-in account without displaying the token:

   ```bash
   uv run hf auth whoami
   ```
6. Rerun the local baseline command. MLX-LM will use the Hugging Face credentials
   stored in the local Hugging Face cache.

See the official [Hugging Face authentication guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
and [MLX-LM project documentation](https://github.com/ml-explore/mlx-lm) for current
details. Never put a Hugging Face token in the repository.

### 3. Confirm the intentional bug

Run the `payment_bug` mini-repository independently:

```bash
uv run pytest test_cases/payment_bug
```

This command is **expected to fail** because the mini-repository is intentionally
buggy. A failure such as `assert 100 == 80.0` confirms that the evaluation case is
working.

### 4. Run the AI debugging baseline

Interactive class command:

```bash
uv run python src/baseline.py payment_bug
```

Select Groq and enter a Top-K value when prompted. For `payment_bug`, Top-K 3 gives
the model the test, checkout implementation, and discount helper.

Reproducible non-interactive command:

```bash
uv run python src/baseline.py payment_bug \
    --provider groq \
    --top-k 3
```

The short name `payment_bug` resolves to `test_cases/payment_bug`. An explicit path
also works:

```bash
uv run python src/baseline.py test_cases/payment_bug \
    --provider groq \
    --top-k 3
```

The result appears in the terminal with the model response revealed in small,
paced chunks. It includes retrieved chunks, diagnosis, evidence, a line-level
suggested change, token usage, and total execution time.

### 5. Run the baseline's own tests

```bash
uv run pytest
uv run ruff check .
```

Root-level pytest collects only `tests/`. It does not accidentally collect the
intentionally failing projects under `test_cases/`.

## Section 1. Problem Definition

### Task

The system must diagnose a reported behavioral bug in a small Python repository.
Given a developer-written bug report and repository source/tests, it should select
relevant code, explain the likely root cause, identify the likely file to change,
and propose a concise replacement without modifying the repository.

### Intended user and situation

The intended user is a developer who has an observable failure but does not yet
know which parts of the repository explain it. The developer wants a grounded
starting point for investigation rather than an automatically applied patch.

### Input

Each input is one self-contained directory under `test_cases/` containing:

- a non-empty `bug_report.txt` written from the user/developer perspective,
- Python implementation files,
- and at least one pytest test that exposes the intentional bug.

For example, `test_cases/payment_bug/bug_report.txt` reports that
`checkout(100, 0.20)` returns `100` instead of `80.0`.

### Expected output

The baseline should print:

- the retrieval method and effective Top-K,
- the repository-relative paths and line ranges retrieved,
- evidence grounded in those chunks,
- a likely root-cause conclusion,
- the repository-relative file implicated by the diagnosis,
- a suggested line-level code change and explanation,
- token usage and measured execution time.

### Operational success and failure criteria

A run succeeds on a test case when:

1. the retrieved context contains the implementation needed to explain the bug;
2. the diagnosis identifies the intended faulty file and behavior;
3. the suggested change is consistent with the expected behavior in the failing
   test; and
4. all required output sections are present and grounded in retrieved code.

A run fails when it retrieves only irrelevant or insufficient context, identifies
the wrong file or behavior, invents code not supported by the repository, suggests
a change that would not satisfy the failing test, or returns an incomplete result.
The CLI labels a structurally incomplete model response as `Diagnosis incomplete`.

## Section 2. Motivation and Project Scope

Software debugging often requires finding a small number of relevant definitions
and tests inside a much larger repository. Sending every file to a model is costly,
may exceed the context window, and can distract the model with irrelevant code.
Retrieval-augmented debugging provides a practical way to study which context helps
an AI system form a correct diagnosis.

Debugging also creates information needs gradually. A test may expose an identifier;
that identifier may lead to a caller; the caller may reveal a configuration value
or state mutation that requires another targeted search. An adaptive system is
therefore a reasonable approach, but useful adaptation must manage context rather
than continuously append everything it finds.

### Project goal

The final project goal is to build and evaluate an **adaptive context-management
system for LLM-based software debugging**. Three capabilities are central:

1. **Dynamic/adaptive retrieval:** decide when the current evidence is insufficient
   and refine retrieval using model findings, identifiers, tests, stack traces, or
   tool observations.
2. **Context compression and pruning:** remove irrelevant or redundant evidence,
   preserve important relationships, and summarize prior observations when raw
   content is no longer needed.
3. **Token-efficient context collection and management:** enforce a controlled
   context budget and prioritize evidence by current debugging relevance instead
   of treating a larger prompt as an automatic improvement.

The goal is not to retrieve indefinitely. It is to decide what to retrieve, when
more information is necessary, what previously collected evidence should remain,
what can be removed or compressed, and what should actually be included in each
LLM prompt.

### Research question

> How can dynamic context retrieval, selection, and compression improve the
> accuracy and token efficiency of LLM-based software debugging compared with
> fixed Top-K retrieval?

This is an empirical question. The project does not assume that additional
retrieval or compression will improve results; evaluation must determine whether
any debugging gains justify the extra complexity, calls, and latency.

### Future architecture

```text
Bug Report
    ↓
Initial Retrieval
    ↓
Context Selection
    ↓
Prune / Compress Context
    ↓
LLM Analysis
    ↓
Determine Missing Information
    ↓
Refine Retrieval Query
    ↓
Retrieve Additional Evidence
    ↓
Merge / Deduplicate / Compress
    ↓
Continue Analysis
    ↓
Diagnosis / Suggested Fix
    ↓
Optional Verification
```

This diagram describes future work, not the current implementation. A future run
may collect more repository evidence overall while sending fewer, more focused
tokens to each model call.

### Context compression

Repository retrieval can collect more information than can or should be sent
directly to the LLM. Future context-management experiments may evaluate:

- removing irrelevant chunks;
- deduplicating overlapping code;
- selecting relevant functions or classes instead of complete files;
- retaining important identifiers, callers, dependencies, and relationships;
- summarizing previously observed evidence;
- preserving test failures and stack traces while discarding unrelated output;
- ranking collected evidence by current debugging relevance; and
- enforcing an explicit context-token budget.

No context-compression method is implemented in the current baseline. These are
candidate strategies to compare experimentally rather than established project
results.

### Token-efficient context management

Token usage is a primary research objective, not only an operational statistic.
The project will distinguish between:

```text
retrieved context
```

which is all evidence collected during retrieval, and:

```text
context actually included in the LLM prompt
```

which is the selected, deduplicated, pruned, or compressed working context sent to
the model. A future adaptive system may retrieve additional evidence over several
steps yet keep individual prompts smaller than a naive append-only approach.

Measurements should include total chunks retrieved, chunks sent to the model,
input tokens per call, total input tokens across the session, output tokens,
retrieval calls, LLM and tool calls, latency, and debugging accuracy.

### Implemented now versus future work

| Implemented now: Static RAG baseline | Future adaptive system |
| --- | --- |
| One retrieval using the original bug report | Iterative retrieval and query refinement |
| Fixed Top-K context | Relevance- and token-budget-aware selection |
| Retrieved chunks are placed directly in one prompt | Pruning, deduplication, and compression |
| One LLM diagnosis | Multiple analysis steps when evidence is missing |
| Advisory suggested change | Targeted tool/test execution on a safe working copy |
| Existing deterministic evaluation test cases | Optional verification loops |

### Scope boundaries

The semester scope includes Python repositories represented by small, independently
runnable test cases, the runnable Static RAG comparison baseline, an adaptive
context-management prototype, and controlled evaluation of accuracy, completeness,
context size, token usage, calls, and latency.

Production-scale indexing, broad multi-language support, automatic modification of
the user's working repository, autonomous multi-agent coordination, unsupervised
deployment, and security guarantees remain out of scope. This keeps the project
small enough to build and evaluate while preserving a meaningful research question.

## Section 3. Baseline System

The baseline is a static, single-pass RAG workflow:

```text
Bug Report
    ↓
Single Retrieval
    ↓
Fixed Top-K Context
    ↓
LLM
    ↓
Diagnosis + Suggested Change
```

The repository is chunked and indexed with local embeddings in ChromaDB before the
single retrieval. The retrieval query is the original bug report. Selected chunks
are sent directly to one LLM request; there is no feedback loop between diagnosis
and retrieval.

### Models, tools, and libraries

- **ChromaDB** stores embedded chunks and performs similarity search.
- **all-MiniLM-L6-v2** is Chroma's lightweight local default embedding model. It
  requires no paid embedding API.
- **Groq with `openai/gpt-oss-20b`** is the primary diagnosis provider.
- **MLX** provides an optional local model path on supported Apple Silicon Macs.
- **Rich** and **Questionary** provide terminal rendering and interactive choices.
- **pytest** and **Ruff** provide tests and linting.
- **uv** manages the locked Python environment.

The embedding model is general-purpose rather than code-specialized. That keeps
the baseline practical to install, while leaving code-aware retrieval as a clear
future comparison.

### Step-by-step behavior

1. Resolve a short name such as `payment_bug` to a directory under `test_cases/`.
2. Read and validate `bug_report.txt`.
3. Recursively find Python source and test files while excluding environments,
   caches, Git data, generated code, hidden files, and local Chroma data.
4. Split files into deterministic overlapping chunks and retain source paths and
   line ranges.
5. Embed and index the chunks in a repository-specific Chroma collection.
6. Embed the bug report and perform one similarity query.
7. Select at most the requested Top-K chunks. If Top-K exceeds the available chunk
   count, report the adjustment and use all available chunks.
8. Build one debugging prompt from the bug report and selected chunks.
9. Ask the configured model for evidence, conclusion, relevant file, suggested
   change, and explanation.
10. Render the advisory result with token usage and actual execution time.

Retrieval does not repeat after the model responds. The model cannot request more
context, modify Top-K, prune or compress selected chunks, apply a patch, run a tool,
or verify its suggestion. This intentional simplicity provides the comparison point
for the future adaptive system.

### Implementation files

```text
src/baseline.py             executable entry point
src/cli.py                  arguments and interactive provider/Top-K selection
src/pipeline.py             single-pass workflow orchestration and timings
src/retrieval.py            repository chunking, models, and interfaces
src/semantic_retrieval.py   ChromaDB indexing and similarity search
src/context_builder.py      debugging prompt construction
src/debugging_result.py     response parsing and source-change analysis
src/terminal_ui.py          terminal presentation and paced response streaming
src/providers/              provider interface, implementations, and factory
```

The components use focused classes and injected collaborators so retrieval, model
access, orchestration, and presentation can be tested independently.

### Chroma persistence and isolation

Chroma collections are stored in the ignored `.chroma/` directory. Each test-case
path is hashed into a stable collection name. The collection is rebuilt on every
run so deleted or edited chunks cannot remain stale. Each record contains raw code,
relative file path, chunk index and identifier, file type, and line range.

This is a reasonable baseline because it is runnable, grounded in repository code,
simple enough to understand, and deliberately lacks the adaptive behavior the
future system is expected to improve.

## Section 4. Sample Input and Baseline Result

### Sample input

`test_cases/payment_bug/bug_report.txt`:

```text
Customers with a 20% discount are being charged the original price.
checkout(100, 0.20) should return 80.0, but currently returns 100.
```

The expected behavior is for the system to connect the failing assertion with
`checkout()` and the helper that calculates the discounted value.

### Observed baseline output

One warm Groq run with Top-K 3 produced:

```text
Retrieval: ChromaDB semantic search · top-k = 3

● Retrieved 3 relevant chunks
  ├─ test_payment.py:1-5
  ├─ payment.py:1-6
  └─ discount.py:1-2

● Diagnosis points to payment.py

  Evidence
  ├─ test_checkout_with_discount asserts checkout(100, 0.20) == 80.0
  ├─ checkout calls apply_discount(...) but ignores its return value
  ├─ apply_discount returns price * (1 - discount)
  └─ checkout returns the original price unchanged

  Suggested change

  5 │     apply_discount(price, discount)
  6 │     return price
  5 └────→     return apply_discount(price, discount)

  469 input · 284 output · 753 total tokens
  Completed in 5.9s
```

This run succeeded because all three relevant files were retrieved and the proposed
change matched the behavior asserted by the test. In contrast, a Top-K 1 run
retrieved only `test_payment.py`; the model lacked the implementation needed for a
grounded file-and-fix suggestion. The terminal now reports that case as
`Diagnosis incomplete` rather than displaying partial output as a valid diagnosis.

## Section 5. Reproducibility and Run Instructions

### Standard reproducible run

From the repository root:

```bash
uv sync --dev

set -a
source .env
set +a

uv run python src/baseline.py payment_bug \
    --provider groq \
    --model openai/gpt-oss-20b \
    --top-k 3
```

Input is read from `test_cases/payment_bug/`, and output is printed to the current
terminal. The command does not modify the test case.

### Available evaluation cases

| Test case | Main bug category | Run the intentional failure |
| --- | --- | --- |
| `payment_bug` | ignored return value/business logic | `uv run pytest test_cases/payment_bug` |
| `auth_bug` | incorrect conditional logic across files | `uv run pytest test_cases/auth_bug` |
| `batching_bug` | exact-boundary edge case | `uv run pytest test_cases/batching_bug` |
| `config_bug` | configuration/default interaction | `uv run pytest test_cases/config_bug` |
| `data_parsing_bug` | quoted CSV parsing | `uv run pytest test_cases/data_parsing_bug` |
| `state_mutation_bug` | unintended state mutation across files | `uv run pytest test_cases/state_mutation_bug` |

Run the baseline on any case by passing its short name:

```bash
uv run python src/baseline.py auth_bug --provider groq --top-k 3
```

### Known setup limitations

- The first semantic run downloads the embedding model and may take longer.
- Groq runs require both internet access and `GROQ_API_KEY`.
- Hosted model output can vary even though the prompt uses deterministic settings.
- Local MLX inference is available only on supported Apple Silicon environments.
- `.chroma/` is generated locally and must remain ignored by Git.
- A very small Top-K may omit the implementation needed for diagnosis.

## Section 6. Initial Evaluation Plan

The future adaptive context-management system will be evaluated against the Static
RAG baseline using the same test cases, initial bug reports, and model configuration
where practical. Neither system should receive credit merely for retrieving or
sending more context.

### Primary evaluation criteria

1. **Retrieval success:** whether collected evidence contains the intended faulty
   file and the supporting code needed to explain it.
2. **Faulty-file identification:** whether the diagnosis points to the intended
   repository-relative file.
3. **Diagnosis accuracy:** whether the explanation identifies the behavior that
   causes the observed failure.
4. **Suggested-change correctness:** whether the proposed change is consistent with
   the failing test and avoids unrelated modifications.
5. **Groundedness:** whether claims and suggested code are supported by repository
   evidence or explicit tool observations.
6. **Output completeness:** whether evidence, conclusion, relevant file, suggested
   change, and explanation are present.
7. **Retrieved-context size:** chunks and estimated tokens collected across all
   retrieval calls before context management.
8. **Final context size:** chunks and tokens actually included in each LLM prompt.
9. **Context compression ratio:** how much collected context remains after
   selection, pruning, deduplication, or compression.
10. **Token usage:** input tokens per model call, total input tokens across the
    debugging session, and output tokens.
11. **Calls and latency:** number of retrieval calls, LLM calls, tool calls, and
    end-to-end execution time.
12. **Reliability:** incomplete responses, failed retrieval/tool calls, and success
    consistency across repeated runs.

The conceptual context compression ratio is:

```text
tokens retained in final context
--------------------------------
tokens collected before pruning
```

A lower ratio is not automatically better: removing evidence that is necessary for
a correct diagnosis is a failure. Compression must be interpreted together with
accuracy, completeness, and groundedness.

For each test case, the intended faulty file and expected behavior provide a small
gold reference. Exact file matching can be combined with a short human rubric for
root-cause, groundedness, and suggested-change correctness. Static RAG should be run
at fixed Top-K values, including a deliberately constrained setting, to identify
cases where its one-shot context is insufficient.

The evaluation should record both all context retrieved during the session and the
smaller context actually sent in each prompt. This distinction allows a future
system to search broadly while remaining token-efficient through selection and
compression.

Possible evidence of improvement would be more correct, complete, grounded
diagnoses at equal or lower total context-token use, or meaningful accuracy gains
with a justified increase in calls and latency. Those outcomes are hypotheses, not
established results. The experiments must determine whether adaptive retrieval and
compression provide enough debugging value to justify their complexity and cost.

## Section 7. Limitations and Next Steps

### Current limitations

- Retrieval happens exactly once using only the original natural-language report.
- Fixed line-window chunking does not understand Python syntax or call graphs.
- The general-purpose embedding model may miss identifiers and subtle code
  relationships.
- Top-K 1 can retrieve a test without the implementation it exercises.
- Rebuilding each collection favors correctness over large-repository speed.
- There is no explicit prompt token-budget enforcement beyond Top-K.
- The model can return incomplete or malformed structured output.
- Suggested code is advisory and is not automatically executed or verified.
- Hosted model availability, latency, and output variation are external risks.

None of the capabilities below are implemented in the current baseline. They define
the planned next phase.

### Adaptive Retrieval

Retrieve additional repository evidence when the current context is insufficient.
New queries may use model findings, failing tests, stack traces, identifiers, call
relationships, and tool observations. The system must decide when another retrieval
is useful rather than retrieve continuously.

### Context Compression

Prune irrelevant chunks, deduplicate overlaps, select relevant definitions, or
summarize accumulated observations so the working context remains focused. Important
tests, failures, identifiers, and relationships should survive compression when
they remain relevant to the current hypothesis.

### Token-Budget Management

Maintain an explicit context-token budget across the debugging session. Evidence
should be ranked and retained by debugging relevance instead of simply increasing
Top-K or appending every retrieved chunk to later prompts.

### Verification

Potentially run targeted tests or other tools against a safe working copy to check
whether a proposed change addresses the observed failure. Verification should be
reported separately from an unverified model suggestion and should not modify the
user's original test case.

The central risk is that retrieval, compression, and verification may add latency,
calls, and failure modes without improving diagnosis quality. The final comparison
must measure whether adaptive, token-efficient context management delivers enough
benefit to justify that additional complexity.

## Generative AI Use

Codex was used to assist with implementation and documentation. Generated changes
were reviewed and tested by the author.
