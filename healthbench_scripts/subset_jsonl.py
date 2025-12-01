#!/usr/bin/env python3
"""
Create a smaller JSONL file by selecting only a subset of entries.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "healthbench" / "2025-05-07-06-14-12_oss_eval.jsonl"
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name(f"{DEFAULT_INPUT.stem}_subset.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a subset of a JSONL file to a new JSONL file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the source JSONL file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the subset JSONL (default: {DEFAULT_OUTPUT})",
    )

    selection = parser.add_argument_group("selection")
    selection.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of initial records to skip before selecting (default: 0)",
    )
    selection.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Keep every Nth record after the offset (default: 1)",
    )
    selection.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after writing this many records (default: keep all eligible records)",
    )
    selection.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, randomly sample this many records (after offset/stride) instead of taking the first ones",
    )
    selection.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used with --sample (default: 0)",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open() as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            yield line


def write_stream_subset(
    input_path: Path,
    output_path: Path,
    offset: int,
    stride: int,
    limit: int | None,
) -> int:
    written = 0
    with input_path.open() as infile, output_path.open("w") as outfile:
        for idx, line in enumerate(infile):
            if not line.strip():
                continue
            if idx < offset or (idx - offset) % stride != 0:
                continue
            outfile.write(line.rstrip("\n") + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    return written


def write_sample_subset(
    input_path: Path,
    output_path: Path,
    offset: int,
    stride: int,
    sample_size: int,
    seed: int,
) -> int:
    records = list(iter_jsonl(input_path))
    eligible_indices = list(range(offset, len(records), stride))
    rng = random.Random(seed)
    rng.shuffle(eligible_indices)
    selected_indices = sorted(eligible_indices[:sample_size])

    with output_path.open("w") as outfile:
        for idx in selected_indices:
            outfile.write(records[idx] + "\n")
    return len(selected_indices)


def main() -> None:
    args = parse_args()
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided")
    if args.sample is not None and args.sample <= 0:
        raise ValueError("--sample must be a positive integer when provided")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.sample is not None:
        written = write_sample_subset(
            args.input,
            args.output,
            offset=args.offset,
            stride=args.stride,
            sample_size=args.sample,
            seed=args.seed,
        )
    else:
        written = write_stream_subset(
            args.input,
            args.output,
            offset=args.offset,
            stride=args.stride,
            limit=args.limit,
        )

    print(json.dumps({"input": str(args.input), "output": str(args.output), "written": written}))


if __name__ == "__main__":
    main()
