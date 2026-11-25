# Data preparation

## Paired retrieval data

The model pair is Qwen2.5-3B-Instruct for the target and Qwen2.5-0.5B-Instruct for the draft. Both revisions are pinned in the [model pair config](../configs/model-pair.json). The 3B checkpoint is about 6.2 GB in BF16 before inference overhead. Use the 4 GB laptop for data preparation and draft-only smoke runs if the draft fits. Use a larger GPU for joint inference and distillation. [Target model](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) · [Draft model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

Run from the repository root in a Python environment with `transformers` and `huggingface_hub`. The laptop’s existing Anaconda environment is `D:/Programming/python.exe`.

```powershell
# Cache and compare model configs, vocabulary, tokenizer behavior, and the chat template.
python -m models.prepare_models

# Use cached tokenizers to generate the pilot JSONL and split manifest.
python -m data.generate_retrieval --offline

# Download draft weights on the laptop, or use --weights both on the training host.
python -m models.prepare_models --weights draft

# Verify the cached files for both models when the full pair is available.
python -m models.prepare_models --weights both --offline
```

The downloader stores pinned snapshots in the ignored `hf_cache/` directory. The generator writes `paired-retrieval.jsonl` and `manifest.json` to `data/generated/paired-retrieval/` by default. The manifest records seed ranges, split counts, length/distance cells, and the JSONL hash. There are 10 test families, four validation families, and four train families across eight cells. The train split supports a small smoke run. For the mismatch development pilot, select validation rows. Keep the test rows for the held-out handoff.

Each `prompt` is the fully formatted text up to the first answer prediction. To recover `input_ids`, encode it with `add_special_tokens=False`. Use `pair_id` to join variants. The evidence end index is exclusive. Use the measured length and distance fields for analysis. Targets are rounded to whole distractor records and checked within 24 tokens. Within each family and target length, the distractor pool and order are fixed across evidence distances.

Use the record schema in the [experiment protocol](../docs/experiment-protocol.md). Follow the synthetic pilot with independent held-out subsets from PG-19 and UltraChat. LongBench and NoLiMa are optional extensions. [DeepMind PG-19](https://huggingface.co/datasets/deepmind/pg19) · [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

Public dataset metadata was verified for PG-19 at `4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b` and UltraChat 200k at `8049631c405ae6576f93f445c6b8166f76f5505a`. Record the exact upstream split and source IDs when extracting subsets.

Place downloaded data in `raw/`, prepared subsets in `processed/`, synthetic prompts in `generated/`, and sensitivity caches in `scores/`. These directories are ignored by Git. Commit preparation code and provenance/split manifests rather than dataset contents.
