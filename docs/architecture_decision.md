# Architecture Decision Memo

## Decision

Ship **Architecture A (single agent)** as the production/MVP default today.
Keep Architecture B available for cases where a second semantic challenge is useful.

## Evidence

Both architectures ran the same six public cases. Existing local result CSVs
support the comparison below; they are generated, ignored artifacts. Manual
validation and offline failure tests supplement the public minimum checks.

| Metric | Single agent | Staged / 2-agent |
|---|---:|---:|
| Cases passing your quality criteria | 6/6 public minimum checks | 6/6 public minimum checks |
| Avg latency | ~0.80 s | ~1.05 s |
| Avg LLM calls | 0.33 | 0.67 |
| Avg tool calls | 6 | 6 |
| Notable policy/grounding failures | None in public minimum checks; deterministic/manual-review degradation tested | None in public minimum checks; another model stage/failure surface |

A used two total model calls and B four. Both gather six evidence tools once,
apply deterministic policy before reasoning, and skip models when clarification
or manual review is already established. Manual checks preserved PII routing,
expired/conflicting vendor handoffs, and legitimate software overlap. A generic
vendor-only false-positive fix was revalidated on training and software requests.

## Trade-offs

B adds an independent reviewer role that can challenge unsupported fit/gap
judgments. It approximately doubles model calls on semantic cases and adds
latency, orchestration complexity and another failure surface without a measured
public quality gain. This does not demonstrate worse quality for B.

A's earlier model-led loop averaged 7.17 HTTP attempts and ~46.85 seconds per
request. Evidence-first orchestration reduced those to 0.33 calls and ~0.80 seconds:
roughly 95% fewer calls and 98% lower latency. One representative optimized call
used 484 tokens; the public run stayed below the tested 8K TPM limit.

## Risks / limitations

Only six public cases and synthetic data were evaluated; minimum checks are not
a full semantic-quality benchmark. Hidden cases and provider behavior can differ.
Both staged roles use the same configured model. Injection detection is heuristic;
safety primarily comes from deterministic enforcement. Overlap uses exact
product/category candidate matching, not fuzzy or semantic equivalence.
Before production, validate broader semantic quality, adversarial inputs,
real-world data, model compatibility and operational failure handling.

## Why this is the right MVP

A provides grounded evidence, mandatory approval routing and safe handoffs with
one optional compact interpretation. Host-owned controls, low reasoning effort,
256-token outputs and no recommendation retries make it defensible within
free-tier limits. Its simpler orchestration is sufficient for the demonstrated
client workflow while leaving B available for targeted review.
