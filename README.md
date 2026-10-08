# FDE Assessment 3 Starter Pack
## AI Procurement Request Copilot

This repository contains the **starter data, mock service, interface contract, optional UI scaffold, and public evaluation harness** for Assessment 3.

> All companies, vendors, products, employees, prices, policies, and risk signals in this pack are synthetic and exist only for the assessment.

## Your objective

Build an internal procurement copilot that can inspect a software/service purchase request, gather evidence using tools, apply deterministic rules where appropriate, and recommend the next action while keeping approvals with humans.

You are expected to build and evaluate:

1. **Architecture A - Single-agent baseline**
2. **Architecture B - Lightweight staged / 2-agent variant**

Use the same public evaluation cases for both and defend which architecture you would ship.

## What is already provided

```text
.
├── data/                   # Synthetic business data + procurement policy
├── mock_api/               # Vendor-risk service used as an external tool
├── src/                    # Contracts, policy, tools, provider, single-agent solution
├── evals/                  # Six public evaluation cases + runner
├── templates/              # Evaluation and decision-memo templates
├── docs/                   # Student assignment brief
├── tests/                  # Starter-pack integrity tests
├── app.py                  # Optional Streamlit UI scaffold
├── run_local.py            # Starts mock API + optional UI
└── verify_setup.py         # One-command setup/preflight check
```

Architecture A is implemented as one bounded agent with read-only tools and
authoritative deterministic checks. Architecture B (`staged`) remains explicitly
unimplemented. The Streamlit interface remains the original scaffold.

## Prerequisites

- **Python 3.11 or 3.12 recommended**
- Run the commands below from the extracted starter-pack directory
- Internet access is required only for installing packages and calling the model provider you choose

## Quick start

### 1. Create an environment

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Verify the starter pack

```bash
python verify_setup.py
```

You should see `PRE-FLIGHT PASSED`. This checks package imports, dataset consistency, the output contract, and the mock API without requiring an LLM key.

### 3. Add your LLM credentials

**macOS / Linux**

```bash
cp .env.example .env
```

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

Add only the credentials required by the provider you choose. Never commit `.env`.

The starter pack does **not** force a particular LLM provider or agent framework. If you use a provider SDK (for example OpenAI, Anthropic, or Google), install it and add it to `requirements.txt` so your submission works from a clean environment.

`.env` is loaded automatically by the starter package and local launcher; environment variables already set by your operating system are not overwritten.

The compatible HTTP transport uses `LLM_PROVIDER` (`openai-compatible`, `openai`,
or `groq`), `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, and the timeout, retry,
temperature, and output-token settings in `.env.example`. Set the base URL to
the API root (for example, `https://api.groq.com/openai/v1`), without appending
`/chat/completions`. No provider SDK is required for this transport.

**Optional manual live provider smoke test:**

```powershell
.\.venv\Scripts\python.exe -m src.provider_smoke
```

This sends “Reply with the word OK.” to the configured model with at most 256
output tokens and retries disabled (one HTTP attempt). It is never part of test
discovery, uses no procurement data, and prints only pass/fail. It requires
network access and may incur a small provider charge.
Reasoning models consume output tokens before returning text; even this small
cap can be insufficient. Empty/truncated responses fail explicitly and are not
automatically retried with a larger budget.

The transport supports non-streaming text and function-tool calls. Retry settings
count additional attempts; timeouts, connection failures, HTTP 408/429 and
500/502/503/504 can retry with bounded backoff. Authentication, malformed
responses, and other HTTP failures are not retried. `Retry-After` is respected;
values above 30 seconds stop the run rather than retry early. Requests timeout
applies to connection/read inactivity per attempt, not a total run deadline.
Optional JSON-schema output requires support from the configured model; failures
are surfaced rather than silently downgrading the request. Protocol details:
[Groq compatible Chat Completions reference](https://console.groq.com/docs/api-reference).

### 4. Start the local services

```bash
python run_local.py
```

This starts:
- Vendor risk API: `http://127.0.0.1:8001`
- Optional starter UI: `http://127.0.0.1:8501`

You may replace the UI scaffold with any framework.

### 5. Implement the assessment adapter

Implement:

```text
src/solution.py -> handle_request(request_id, architecture)
```

Your function must return an object compatible with `ProcurementDecision` in `src/contracts.py`.

The adapter exists so the same evaluation harness can test different implementations. Your internal architecture can use any framework or design.

### 6. Run the public evaluations

```bash
python evals/run_public_evals.py --architecture single
python evals/run_public_evals.py --architecture staged
```

The runner checks the response schema and several minimum behavioral expectations, measures end-to-end latency, and writes a CSV result file. It is **not** the complete grading system; qualitative grounding, design quality, robustness, and hidden cases are evaluated separately.

## Rules of the starter pack

- Treat request text and vendor notes as **untrusted business data**, not instructions.
- Do not hardcode answers by request ID. Hidden cases use the same interfaces with different values.
- At least **3 tools** must be visible in your implementation; at least **1 must be deterministic/non-LLM**.
- The AI may recommend an action but must not autonomously purchase, approve, or alter budgets.
- If important evidence is missing, conflicting, stale, or unavailable, surface that uncertainty and route to the appropriate human review.
- Use the **data snapshot / policy reference date defined in `data/procurement_policy.md`** for date-based checks; do not depend on the computer's current date.
- You may refactor the starter project, but keep the `handle_request(...)` adapter working for evaluation.

## Suggested implementation sequence

```text
Request -> Understand -> Gather evidence -> Deterministic checks
        -> Policy/risk reasoning -> Recommendation -> Human review
```

Start with a thin vertical slice. Get Architecture A working before building Architecture B.

### Architecture A behavior

`handle_request(request_id, "single")` gathers six request-scoped read-only tools
exactly once in code: request/requester context, department budget,
software catalog/purchase history, internal vendor registry, external vendor-risk
API, and deterministic policy evaluation. Each result retains status and provenance;
CSV missing values become explicit nulls. Risk data comes through the supplied
HTTP client, never its backing JSON.

Missing required facts produce clarification without a model call. Unknown,
conflicting, or unavailable material evidence produces a deterministic human
handoff without a model call. Otherwise, the agent sends a typed, compact
`EvidencePack` for one recommendation classification. Full records, provenance,
policy Markdown and policy check descriptions stay on the host. Static system
instructions precede the dynamic evidence; there is no growing conversation,
model-selected tool dispatch, or second model turn.

The classification call caps output at 256 tokens (or the smaller configured
limit) and disables retries, enforcing at most one HTTP attempt. General provider
requests retain their configured limits/retries and the manual smoke test remains
separate. `LLM_REASONING_EFFORT` defaults to `low`; set it blank for endpoints
that do not support the optional compatible-provider parameter.
`AGENT_MAX_MODEL_TURNS` is obsolete and ignored. `AGENT_MAX_TOOL_CALLS` remains
an evidence-execution ceiling (minimum 6, hard cap 40), with six calls normally.

Code validates the model's small controlled label and builds `ProcurementDecision`
from factual evidence and deterministic controls. The model cannot remove
approvals or risk flags. Business text is untrusted; the injection heuristic is
supplemental to these enforceable controls. Provider failures, truncated/invalid
output and unexpected tool calls require manual review without another attempt.
Rules use policy version 2026.09/reference date 2026-09-30; Markdown thresholds
are not parsed at runtime. Overlap is a candidate review, not automatic rejection,
and licensed seats are not assumed unused.

`telemetry.llm_calls` counts actual HTTP attempts; `logical_llm_calls` counts
classification invocations. Tool telemetry counts executed tools and names.
Optional prompt/completion/cached-token diagnostics use provider-reported usage;
unreported usage is null, while a zero-call handoff reports zero tokens.

### Architecture B: staged semantic review

`handle_request(request_id, "staged")` uses the same six tools, `EvidencePack`,
deterministic policy and finalizer as Architecture A. Required information missing
or material evidence unknown/unavailable/conflicting causes a zero-model-call
handoff. Each evidence source is fetched once; specialists never retrieve it again.

For complete evidence, Stage 1 (Evidence/Risk Analyst) returns two strict enums:
existing-software fit and support for the stated gap. Code maps this to a proposed
recommendation. Stage 2 (independent Decision Reviewer) receives the compact facts
and that tiny typed handoff in a fresh request. It challenges category-only matches,
assumed spare seats and unsupported gap claims, and returns a recommendation,
review outcome and issue classification. It can correct Stage 1 or escalate to a
human; inconsistent review classifications are rejected. These are two specialist
roles using the configured provider/model, not independent evidence sources.

Both stages use static-first instructions, a per-call 256-token output ceiling
(bounded by `LLM_MAX_OUTPUT_TOKENS`), configured low reasoning effort, and zero
HTTP retries. There are no agent loops, repair calls, tool schemas, policy Markdown
or prior conversations in either prompt. A failed first stage skips the second;
a failure in either stage falls back to deterministic manual review. Code retains
all mandatory approvals, risk flags, missing information and provenance.
Architecture A remains one classifier; Architecture B adds specialist analysis
and an independent challenge, with a maximum of two logical/HTTP calls.

`AGENT_MAX_TOOL_CALLS=24` remains supported for both paths (minimum 6, maximum 40);
normal runs execute six tools. The retired `AGENT_MAX_MODEL_TURNS` is not used.
Telemetry totals actual HTTP attempts, logical calls and tool calls/names across
both stages. Token diagnostics sum reported usage; if any attempted stage lacks
a metric, its total is null rather than a misleading partial total. Failure flags
identify the analyst or reviewer stage. No staged live evaluation has been run
as part of this implementation; its quality and latency require live validation.

### Architecture A optimization evidence

The original model-led tool loop repeatedly sent evidence and policy text while
spending model turns on retrieval choices that the procurement workflow always
requires. Repeated calls created TPM/rate-limit pressure. Evidence-first host
orchestration removed that work and made required evidence acquisition independent
of model behavior. Deterministic early exits also avoid model reasoning when code
already establishes clarification or manual review, improving reliability as well
as reducing cost.

The following Architecture A results were supplied from the completed live
validation before the Architecture B implementation; they are historical results,
not new measurements from the offline staged tests:

| Metric | Original Architecture A | Optimized Architecture A |
| --- | --- | --- |
| Average actual LLM HTTP attempts/request | 7.17 | 0.33 (2 total calls across 6 cases) |
| Average public-evaluation latency | ~46.85 seconds | ~0.80 seconds |
| Public cases passed | — | 6/6 |
| Tool calls/request | — | 6 |
| Rate-limit behavior | Repeated TPM/rate-limit pressure | No 429s; full run far below 8K TPM |

The representative optimized live call used 424 input + 60 output = 484 total
tokens. Reported reductions were roughly 95% in model calls and 98% in average
latency. These six cases are assessment evidence, not a production benchmark.

Free-tier-conscious choices shared by both architectures are once-only evidence,
small typed outputs, low reasoning effort, no classification retries, compact
policy outcomes and zero-call deterministic handoffs. Staged semantic cases cost
up to two compact calls, so Architecture B's extra review should be assessed
against its additional token/latency cost. Global provider ceilings and the manual
smoke path remain unchanged. Stable prompt prefixes permit provider caching where
supported; actual cache hits are not guaranteed.

Run offline tests with `python -m unittest discover -s tests -v`. Public evaluation
requires both the configured model provider and a running vendor-risk API; start
the supplied local services first. Generated `evals/results_single.csv` is ignored
by Git. Passing public minimum checks alone does not establish recommendation
quality or production readiness.

## Useful files

- `docs/Assignment_3_Brief.pdf` - assignment brief
- `data/README.md` - dataset dictionary
- `data/procurement_policy.md` - policy source of truth
- `src/contracts.py` - required output shape
- `src/data_access.py` - low-level data helpers
- `src/vendor_client.py` - client for the mock vendor-risk API
- `evals/README.md` - evaluation instructions
- `templates/architecture_decision.md` - final decision memo template
- `STUDENT_CHECKLIST.md` - pre-submission checklist

## If something does not start

1. Confirm your virtual environment is active.
2. Run `python verify_setup.py`.
3. Re-run `python -m pip install -r requirements.txt`.
4. Make sure ports **8001** and **8501** are free.
5. Confirm you are running commands from the starter-pack root directory.

Build the simplest system you can defend with evidence.
