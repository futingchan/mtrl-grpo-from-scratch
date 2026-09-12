"""Harbor task grader (stdlib only). Prints the scalar reward to stdout."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    answer_path = Path(args.answer)
    if not answer_path.exists():
        print("0.0")
        return
    answer_text = answer_path.read_text()
    gold = json.loads(Path(args.gold).read_text())
    breakdown = score(
        gold["doc"],
        answer_text,
        gold["gold_todos"],
        gold["source_entities"],
    )
    if args.breakdown:
        print(json.dumps(breakdown.to_dict(), indent=2), file=sys.stderr)
    print(f"{breakdown.total:.6f}")


if __name__ == "__main__":
    main()
