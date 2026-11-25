"""Generate paired retrieval prompts with measured token distances and disjoint splits."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from models.prepare_models import ROOT, prepare_models


SPLIT_SEEDS = {
    "train": range(1000, 1004),
    "validation": range(2000, 2004),
    "test": range(3000, 3010),
}
PILOT_CELLS = [
    (1024, 128), (2048, 128), (4096, 128), (8192, 128),
    (8192, 512), (8192, 2048), (8192, 4096), (8192, 7168),
]
TOKEN_TOLERANCE = 24


def nearest_index(measure, stop, goal):
    """Find the closest index in a nondecreasing integer measurement."""
    low, high = 0, stop
    while low < high:
        middle = (low + high) // 2
        if measure(middle) < goal:
            low = middle + 1
        else:
            high = middle
    return min({max(0, low - 1), low}, key=lambda i: (abs(measure(i) - goal), i))


def generate_family(tokenizer, split, seed, cells):
    rng = random.Random(seed)
    limit = max(length for length, _ in cells)
    keys = rng.sample(range(16 ** 6), limit + 1)
    values = rng.sample(range(16 ** 6), limit + 1)
    target_key = f"R{keys[0]:06X}"
    answer = f"V{values[0]:06X}"
    evidence = f"{target_key}: {answer}"
    distractors = [f"R{key:06X}: V{value:06X}\n" for key, value in zip(keys[1:], values[1:])]

    placeholder = "RECORD_LIST_PLACEHOLDER"
    user_message = (
        "Read the records below. Each record maps a key to a value.\n\n"
        f"{placeholder}\nWhat value is associated with {target_key}? Respond with just the value."
    )
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}], tokenize=False, add_generation_prompt=True,
    )
    prefix, suffix = template.split(placeholder)

    def render(count, position):
        before = "".join(distractors[:position])
        after = "".join(distractors[position:count])
        return prefix + before + evidence + "\n" + after + suffix, len(prefix) + len(before)

    def context_size(count):
        prompt, _ = render(count, 0)
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    distractor_counts = {}
    pair_id = f"retrieval-{seed}"
    for length, distance in cells:
        if length not in distractor_counts:
            distractor_counts[length] = nearest_index(context_size, limit, length)
        count = distractor_counts[length]

        def measure(position):
            prompt, evidence_start = render(count, position)
            offsets = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            evidence_end = evidence_start + len(evidence)
            first = next(i for i, (_, end) in enumerate(offsets) if end > evidence_start)
            end = next((i for i in range(first, len(offsets)) if offsets[i][0] >= evidence_end), len(offsets))
            return prompt, len(offsets), first, end

        def distance_from_end(lines_after_evidence):
            _, context_tokens, first, _ = measure(count - lines_after_evidence)
            return context_tokens - first

        lines_after_evidence = nearest_index(distance_from_end, count, distance)
        prompt, context_tokens, evidence_start, evidence_end = measure(count - lines_after_evidence)
        actual_distance = context_tokens - evidence_start
        if abs(context_tokens - length) > TOKEN_TOLERANCE or abs(actual_distance - distance) > TOKEN_TOLERANCE:
            raise ValueError(f"{pair_id}: wanted ({length}, {distance}), got ({context_tokens}, {actual_distance})")

        yield {
            "sample_id": f"{pair_id}-l{length}-d{distance}",
            "pair_id": pair_id,
            "split": split,
            "prompt": prompt,
            "reference_answer": answer,
            "context_tokens": context_tokens,
            "evidence_position": evidence_start,
            "evidence_end_position": evidence_end,
            "dependency_distance": actual_distance,
            "target_context_tokens": length,
            "target_dependency_distance": distance,
            "seed": seed,
            "task_type": "retrieval",
        }


def write_pilot(tokenizer, settings, output_dir):
    records = [row for split, seeds in SPLIT_SEEDS.items() for seed in seeds
               for row in generate_family(tokenizer, split, seed, PILOT_CELLS)]
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired-retrieval.jsonl").write_bytes(data)
    manifest = {
        "dataset": "synthetic-key-value-retrieval-pilot-v1",
        "tokenizer": settings["target"],
        "split_seeds": {split: list(seeds) for split, seeds in SPLIT_SEEDS.items()},
        "counts": dict(Counter(row["split"] for row in records)),
        "cells": PILOT_CELLS,
        "token_tolerance": TOKEN_TOLERANCE,
        "jsonl_sha256": hashlib.sha256(data).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/generated/paired-retrieval")
    parser.add_argument("--offline", action="store_true", help="Use the local workspace cache only")
    args = parser.parse_args()
    tokenizer, settings = prepare_models(offline=args.offline)
    manifest = write_pilot(tokenizer, settings, args.output_dir)
    print(f"Wrote {manifest['counts']} to {args.output_dir}")


if __name__ == "__main__":
    main()
