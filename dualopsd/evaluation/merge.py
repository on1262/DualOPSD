"""Merge the student LoRA adapter of a checkpoint into a standalone Hugging Face model."""

from __future__ import annotations

import argparse
from pathlib import Path

from dualopsd.evaluation import escapes_scratch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dualopsd merge", description=__doc__)
    parser.add_argument("--base", required=True, help="base model path or hub id")
    parser.add_argument("--adapter", required=True, help="adapter dir, e.g. <run>/checkpoint-100/student")
    parser.add_argument("--output", required=True, help="merged model directory (must be empty or absent)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(args.output)
    if escapes_scratch(output):
        raise PermissionError(f"refusing to write through a symlink leaving $DUALOPSD_SCRATCH: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype="auto", device_map="cpu")
    model = PeftModel.from_pretrained(base, args.adapter, adapter_name="student")
    merged = model.merge_and_unload()
    # Do not inherit sampling-only generation defaults; evaluation passes sampling params explicitly.
    merged.generation_config.do_sample = False
    merged.generation_config.temperature = None
    merged.generation_config.top_p = None
    merged.generation_config.top_k = None
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(output)


if __name__ == "__main__":
    main()
