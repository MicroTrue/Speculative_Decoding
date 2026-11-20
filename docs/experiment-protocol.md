# Experiment protocol

This document defines the starting conventions for future implementation. It records proposed procedures, not completed measurements.

## Models and reproducibility

Use one manifest per run, beginning with [the template](../configs/experiment.template.json). Record immutable model/tokenizer revisions, code commit, hardware, precision, framework versions, seeds, chat template, dataset provenance, and checkpoint identifiers. Resolve null settings before collecting results.

Verify tokenizer vocabulary and token-ID compatibility, special tokens, chat formatting, supported context length, and memory use. Compare target and draft distributions over the same vocabulary and identical tokenized prediction prefixes. A shared model family is not itself proof of compatibility.

## Prompt records

Each record should contain:

| Field | Meaning |
| --- | --- |
| `sample_id` | Unique record identifier |
| `split` | Train, validation, or test |
| `prompt` | Exact input text |
| `reference_answer` | Expected retrieval answer where applicable |
| `context_tokens` | Actual token count of the formatted input prefix |
| `evidence_position` | Zero-based token index of the beginning of the relevant evidence span |
| `dependency_distance` | Token distance from evidence start to the first answer prediction boundary |
| `seed` | Data-generation seed |
| `task_type` | Retrieval, continuation, conversation, or other documented workload |

For the initial single-span retrieval task, the prediction boundary is the token count of the final formatted input, and `dependency_distance = context_tokens - evidence_position`. Also record evidence span end positions where available. Document any different convention before combining results.

Compute positions after applying the agreed tokenizer, special tokens, and chat template. Preserve evidence content across paired variants. Keep training, validation, and evaluation examples disjoint, including generated seeds and source documents; retain pair IDs for paired analysis. Locally generated RULER-style prompts must be distinguished from official benchmark evaluation.

## Sensitivity and objectives

Start with forward KL between the full-context and recent-context teacher distributions on identical prediction targets. Candidate local windows are 128, 512, and 1,024 tokens. Compare position-preserving context removal and targeted evidence replacement to diagnose truncation artifacts.

Validate scores against known evidence locations, draft–target disagreement, and actual rejection rates before using them as a learning signal. Fix the normalization/clipping rule using training/validation data and record its parameters; do not tune it on test results.

- M1: conventional forward-KL teacher-to-draft distillation.
- M2: length-balanced distillation with mixed short/long exposure.
- M3: per-token weight `1 + alpha * normalized_clipped_sensitivity`, with weighted KL divided by the sum of valid token weights.
- M4: a recorded combination of short-context, long-context, and dependency-weighted KD objectives.

Mask padding and non-training positions consistently. For M4, define group membership, component normalization, and coefficients before comparing runs. Use matched initialization, optimizer settings, training token budgets, and short/long exposure wherever possible. M2 vs. the proposed method should isolate dependency weighting. Record teacher scoring and additional training costs.

## Decoder correctness

Begin with greedy target-only and greedy assisted generation. Hold prompt tokenization, stopping rules, precision, and generation limits fixed. Check output equivalence and investigate numerical or implementation mismatches before reporting speedups. Checkpoint changes affect only the draft; the target remains fixed across M0–M4.

Define draft length consistently and record early termination. Account separately for proposed draft tokens, accepted draft tokens, rejected proposals, target correction tokens, and target bonus tokens. Corrections and bonus tokens count toward generated output but not accepted draft proposals.

## Metrics and timing

| Metric | Reporting convention |
| --- | --- |
| Acceptance rate | Accepted draft tokens / proposed draft tokens; report both counts |
| Accepted tokens per verification | Accepted draft tokens / verification rounds; report generated tokens per round separately if useful |
| Decode throughput | Generated output tokens / measured decode seconds |
| Time per output token (TPOT) | State the timing interval and denominator; distinguish decode average from inter-token latency |
| End-to-end latency | Report measured request processing time and any excluded stages |
| Speedup | Target-only latency / assisted latency on matched inputs, settings, and measured intervals |
| Peak GPU memory | Record allocated/reserved measurement method and included model/cache state |
| Training cost | GPU-hours, trained token count, scoring cost, and checkpoint identity |

Run warm-up and repeated timed measurements. Synchronize GPU work at timing boundaries; record hardware and competing workloads. Separate prefill from decode, and record output lengths rather than attributing shorter generations to a faster decoder. Record instrumentation overhead where relevant.

Measure teacher retrieval accuracy alongside KL, top-1 agreement, and acceptance. Use paired comparisons and report uncertainty; include all-example results and teacher-correct subsets. Retain short-context metrics and negative results.

## Artifacts and ablations

Keep raw data, weights, checkpoints, and individual run output outside Git. Commit run configurations, curated result tables, and artifact paths or IDs sufficient to reproduce a reported result. Record external dataset provenance and applicable usage terms.

Prioritize removal of dependency weighting, removal of short-context balancing with matched exposure, random weighting, and draft–target disagreement weighting. Explore local windows, objective weights, and speculative draft length only as resources permit. Select settings on validation data.
