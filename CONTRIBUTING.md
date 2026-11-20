# Contributing

This project follows the roles and scientific gates in the supplied two-person research plan. Roles are starting responsibilities, not fixed ownership restrictions.

- **Member A — systems and evaluation:** model loading, instrumented decoding, mismatch analysis, and reproducible inference benchmarks; reviews training protocol.
- **Member B — data and distillation:** controlled prompts, dataset preparation, sensitivity scoring, losses, and training; reviews evaluation protocol.

## Workflow

Use a short-lived feature branch, such as `feat/paired-prompts` or `feat/greedy-decoder`, and open a pull request into `main`. Agree on shared data and metric interfaces before building against them. Ask the other team member to review changes affecting those interfaces.

Keep each change focused on a milestone in [the roadmap](docs/roadmap.md). Describe what works, the checks actually run, and remaining limitations. Planned functionality and proposed settings must be labeled as such.

## Experiment records

Start from [the manifest template](configs/experiment.template.json) and follow [the protocol](docs/experiment-protocol.md). Every reported experiment must identify its code revision, model and tokenizer revisions, dataset version and split, seed, hardware, precision, decoding settings, training budget, and checkpoint or artifact ID.

Use disjoint training, validation, and test data. Select hyperparameters on validation data. Match training tokens and optimizer settings when comparing methods, and record any deviations.

Datasets, weights, checkpoints, tokens, and raw run outputs belong outside Git. Commit curated summaries, configurations, and artifact references. Check `git diff --cached` before publishing.

## Validation

There is no runnable implementation or pinned dependency environment yet. Introduce the environment with the first working model loader. Add checks that establish scientific or implementation correctness, particularly tokenizer compatibility, exact evidence positions, greedy output equivalence, and accepted-token accounting.

For GPU benchmarks, record warm-up, synchronization, repeated timings, prefill/decode separation, and memory measurements. Report failures and negative results alongside improvements.
