# Data preparation

Planned home for paired retrieval prompt generation, dataset preparation, and split manifests. No dataset loader or generator is implemented yet.

Use the record schema in [the experiment protocol](../docs/experiment-protocol.md). Begin with controlled RULER-style prompts, then independent PG-19 and short-context UltraChat subsets. LongBench and NoLiMa are optional extensions.

Place downloaded data in `raw/`, prepared subsets in `processed/`, synthetic prompts in `generated/`, and sensitivity caches in `scores/`. These directories are ignored by Git. Commit preparation code and provenance/split manifests rather than dataset contents.
