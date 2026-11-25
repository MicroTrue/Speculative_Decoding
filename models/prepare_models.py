"""Cache pinned model snapshots and verify the shared token space."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs/model-pair.json"
CACHE_DIR = ROOT / "hf_cache"
METADATA_FILES = [
    "config.json", "generation_config.json", "tokenizer*.json", "vocab.json", "merges.txt", "LICENSE",
]


def prepare_models(weights="none", offline=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    settings = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    tokenizers = {}
    configs = {}
    for role in ("draft", "target"):
        model = settings[role]
        snapshot = CACHE_DIR / f"models--{model['id'].replace('/', '--')}" / "snapshots" / model["revision"]
        files = METADATA_FILES
        if weights == "both" or (weights == "draft" and role == "draft"):
            files = files + ["*.safetensors", "*.safetensors.index.json"]
        print(f"Preparing {role}: {model['id']} @ {model['revision']}", flush=True)
        if not offline:
            snapshot_download(
                repo_id=model["id"], revision=model["revision"], cache_dir=CACHE_DIR / ".cache",
                local_dir=snapshot, allow_patterns=files, token=False,
            )
        if not (snapshot / "tokenizer.json").is_file() or not (snapshot / "config.json").is_file():
            raise FileNotFoundError(f"Incomplete {role} metadata in {snapshot}")
        if "*.safetensors" in files:
            index = snapshot / "model.safetensors.index.json"
            if index.is_file():
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            else:
                shards = {"model.safetensors"}
            missing = sorted(shard for shard in shards if not (snapshot / shard).is_file())
            if missing:
                raise FileNotFoundError(f"Incomplete {role} weights: {', '.join(missing)}")
        tokenizers[role] = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True)
        configs[role] = AutoConfig.from_pretrained(snapshot, local_files_only=True)

    target, draft = tokenizers["target"], tokenizers["draft"]
    if (not target.is_fast or not draft.is_fast
            or target.get_vocab() != draft.get_vocab()
            or target.special_tokens_map != draft.special_tokens_map
            or target.chat_template != draft.chat_template
            or target.backend_tokenizer.to_str() != draft.backend_tokenizer.to_str()):
        raise ValueError("The draft and target do not have identical tokenizers and chat templates")
    if configs["target"].vocab_size != configs["draft"].vocab_size:
        raise ValueError("The draft and target have different model vocabulary dimensions")
    for role, config in configs.items():
        if config.max_position_embeddings <= settings["context_limit"]:
            raise ValueError(f"{role} context limit leaves no room for generation")
    print(f"Verified tokenizer equality and {configs['target'].vocab_size} output logits", flush=True)
    return target, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", choices=("none", "draft", "both"), default="none")
    parser.add_argument("--offline", action="store_true", help="Use the local workspace cache only")
    args = parser.parse_args()
    prepare_models(weights=args.weights, offline=args.offline)


if __name__ == "__main__":
    main()
