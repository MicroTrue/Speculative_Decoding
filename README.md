# Cost-Aware Long-Range Dependency Distillation

An MLSys course research project investigating whether dependency-aware knowledge distillation can make speculative decoding faster on long-context workloads while preserving short-context performance.

**Status:** Model preparation and controlled retrieval data generation are available. Decoder, training, and evaluation code are pending.

## Research question

Can a small draft model learn when a target model relies on distant information, and can that improve speculative decoding enough to outweigh drafting and verification costs?

Long input sequences do not always require long-range reasoning. The project separates **total context length** from **the distance to relevant evidence**, then tests whether selectively emphasizing dependency-sensitive predictions improves draft–target alignment.

The proposed starting signal is the teacher's next-token distribution change when distant context is removed:

```text
s_t = KL(p_target(. | full context) || p_target(. | recent context))
w_t = 1 + alpha * normalized_and_clipped(s_t)
dependency_weighted_KD = sum_t(w_t * KL(p_target,t || p_draft,t)) / sum_t(w_t)
```

Sensitivity is a proposed proxy that needs validation: truncation and positional changes can also alter predictions. The initial method adds training-time supervision; it does not require a new inference-time scoring step.

## Planned comparisons

| Variant | Description | Purpose |
| --- | --- | --- |
| Target-only | Autoregressive generation with the target | Latency and output reference |
| M0 | Original, untuned draft | Speculative decoding baseline |
| M1 | Conventional forward-KL distillation | Standard distillation baseline |
| M2 | Length-balanced distillation | Control for mixed short/long training exposure |
| M3 | Dependency-weighted distillation | Test sensitivity-based token weighting |
| M4 | Balanced short, long, and dependency-weighted objectives | Test short-context retention |

Initial model candidates from the execution plan are **Qwen2.5-3B-Instruct** (target) and **Qwen2.5-0.5B-Instruct** (draft), beginning with batch size 1 and 1K–8K input contexts. Tokenizer compatibility, model revisions, context limits, precision, and memory use must be verified before runs.

## Evaluation

| Workload | Role | Priority |
| --- | --- | --- |
| RULER-style controlled prompts | Separate context-length and evidence-distance effects | Core |
| PG-19 | Sustained natural-text continuation | Core |
| UltraChat | Short-context performance retention | Core |
| LongBench / NoLiMa | Application generalization and harder retrieval | Optional after core experiments |

Record draft–target KL, top-1 agreement, teacher accuracy, proposed/accepted draft tokens, accepted tokens per verification, output throughput, time per output token, end-to-end latency, peak GPU memory, and training GPU-hours. Measure prefill and decode separately. Acceptance improvements alone do not establish a speedup.

## Repository layout

```text
configs/          Shared experiment manifest template
models/           Pinned snapshot preparation and tokenizer compatibility checks
data/             Paired retrieval prompt generator and dataset preparation notes
distillation/     Planned sensitivity scoring, losses, and training
decoding/         Planned target-only and speculative decoding
analysis/         Planned mismatch analysis and plotting
evaluation/       Planned benchmark runners and metrics
experiments/      Planned reproducible launch scripts
results/          Curated summaries and artifact references
tests/            Planned correctness and reproducibility checks
docs/             Milestones and experiment protocol
.github/          Issue and pull request templates
```

Raw data, model weights, checkpoints, caches, and run outputs are excluded from Git.

## Start here

1. Read the [research proposal](Speculative_Decoding_Research_Proposal.docx) and [two-person execution plan](Speculative_Decoding_Two_Person_Research_Plan%281%29.docx).
2. Follow the [milestones](docs/roadmap.md) and agree on the [experiment protocol](docs/experiment-protocol.md). Use the [data preparation commands](data/README.md#paired-retrieval-data) to prepare the pinned snapshots and generate the paired prompts.
3. Copy `configs/experiment.template.json` to a run-specific config and replace unresolved fields before collecting results.
4. Establish G1: compatible models, reproducible tokenized inputs, and a target-only/assisted greedy output check. Run a small mismatch pilot before scaling training.

A Python environment and dependency versions will be pinned when the model-loading implementation is introduced and verified on the available hardware. There is no training or benchmark command yet.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the two-person workflow and experiment reporting requirements.
