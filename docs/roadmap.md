# Research roadmap

Derived from the supplied research proposal and two-person execution plan. Every gate below is currently pending. The original plan proposes four weeks; timing depends on hardware and pilot evidence.

| Stage | Lead | Work | Completion gate |
| --- | --- | --- | --- |
| S1: infrastructure | A: models/decoder; B: datasets | Check model/tokenizer compatibility, prepare paired prompts, establish manifests and greedy output equivalence | G1: compatible models, reproducible inputs, smoke checks |
| S2: characterize mismatch | A, with B validating data | Vary context length independently of evidence distance; measure disagreement and teacher accuracy | G2: length/distance curves and evidence for or against dependency-specific mismatch |
| S3: validate sensitivity | B implements; A validates | Compare full/local teacher distributions; check truncation and position controls | G3: validated sensitivity score and fixed scoring settings |
| S4: baseline distillation | B trains; A evaluates | Preserve M0; train M1 and M2 with recorded budgets | G4 baseline: baseline checkpoints, configurations, costs, and evaluation |
| S5: proposed distillation | B trains; A evaluates | Train M3/M4 after G2/G3, starting with a small validation sweep | G4 complete: matched-budget M0–M4 comparison |
| S6: systems evaluation | A, with checkpoints from B | Evaluate core workloads against target-only decoding on identical hardware/settings | Reproducible throughput, latency, acceptance, memory, and retention table |
| S7: ablations and report | Both | Remove weighting/balancing; compare random and disagreement weighting; explain failure cases | G5: final tables, figures, reproducible artifacts, and report |

## First shared milestone

- A: load the candidate models; verify tokenizer and context compatibility; implement next-token KL/top-1 agreement on identical prefixes; establish target-only greedy generation.
- B: build paired retrieval prompts with fixed-length, variable-distance examples; calculate positions after tokenization; reserve held-out seeds; begin full/local scoring and a tiny KD smoke run.
- Both: agree on schema, revisions, seeds, chat template, precision, and metrics before collecting results.

Mismatch experiments can start before the instrumented decoder is complete. Baseline coding can proceed while sensitivity is being validated. Full proposed-method training depends on the problem characterization and sensitivity gates.

## Initial controlled experiments

1. Vary input length across 1K, 2K, 4K, and 8K tokens while keeping relevant evidence approximately 128 tokens before the prediction.
2. Fix input length at 8K and place identical evidence approximately 128, 512, 2K, 4K, and 7K tokens before the prediction.
3. After a small pilot, evaluate feasible length × distance cells with approximately 50–100 paired prompts per cell.

Treat these as proposed starting settings. Verify actual token lengths and distances. Report paired confidence intervals, teacher accuracy, results on all examples, and the teacher-correct subset.

## Decision points

If dependency-specific mismatch is weak, investigate prompt difficulty and model capacity before scaling training. If the sensitivity score mostly reflects truncation or position artifacts, revise it before M3/M4. If acceptance improves without latency gains, analyze drafting, verification, and memory costs.

For limited GPU memory, start with shorter inputs, sampled answer-token positions, and small LoRA runs. LongBench, NoLiMa, and 16K–32K contexts are not prerequisites for the core result. With one GPU, alternate training and profiling blocks while doing coding and analysis on CPU.
