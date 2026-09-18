"""Legacy training script for the DLA study (imported material, not executed by ResearchOS).

Kept verbatim so that Research Import has something real to reconstruct an experiment from:
an argparse surface, a config file, a seed list, and the metric names that appear in runs/*.csv.
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train one DLA arm.")
    parser.add_argument("--arm", choices=["direct", "shufwrite"], required=True)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--dataset", default="wikitext-103")
    parser.add_argument("--scale", default="124M")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--sleep-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/out.json"))
    return parser.parse_args()


METRICS = ["retention_at_1", "retention_at_4", "loss_final", "param_updates"]


def retention_at(window: int, writes: int) -> float:
    """Fraction of fast weights still readable `window` contexts later."""
    if writes == 0:
        return 0.0
    return min(1.0, 0.42 / (window ** 0.5))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    config = json.loads(args.config.read_text()) if args.config else {}
    result = {
        "arm": args.arm,
        "seed": args.seed,
        "model": args.model,
        "dataset": args.dataset,
        "retention_at_1": retention_at(1, args.steps),
        "retention_at_4": retention_at(4, args.steps),
        "loss_final": 2.9 + random.random() * 0.05,
        "param_updates": args.steps,
        "config": config,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
