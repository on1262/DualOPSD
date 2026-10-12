"""Unified command line: ``python -m dualopsd <command> [args]``."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "train": ("dualopsd.train", "OPSD / DualOPSD distillation"),
    "sft": ("dualopsd.baselines.sft", "SFT baseline (run under accelerate launch)"),
    "grpo": ("dualopsd.baselines.grpo", "GRPO baseline (run under accelerate launch)"),
    "merge": ("dualopsd.evaluation.merge", "merge a student LoRA adapter into the base model"),
    "eval": ("dualopsd.evaluation.benchmark", "AIME24/AIME25/HMMT25 avg@k evaluation"),
    "generate": ("dualopsd.evaluation.generate", "teacher/student generation on the 500-problem set"),
    "judge": ("dualopsd.evaluation.judge", "terminal-answer judge with Qwen3-8B"),
    "prepare-data": ("dualopsd.data.prepare", "build training and evaluation data"),
    "posthoc": ("dualopsd.diagnostics.posthoc", "post-hoc KL / clipping / JS-retention re-scoring"),
    "recurrence": ("dualopsd.diagnostics.recurrence", "clipped-token recurrence R_k"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["usage: python -m dualopsd <command> [args]", "", "commands:"]
    lines += [f"  {name:<{width}}  {description}" for name, (_, description) in COMMANDS.items()]
    lines.append("\nRun `python -m dualopsd <command> --help` for command options.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(usage())
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"unknown command: {command}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[command][0])
    sys.argv = [f"dualopsd {command}", *rest]
    result = module.main(rest)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
