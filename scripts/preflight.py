"""Preflight checks for Tier A training. Exit 1 on hard failure.

Usage:
    python scripts/preflight.py            # strict: needs a working CUDA GPU
    python scripts/preflight.py --cpu-ok   # GPU checks become warnings
    python scripts/preflight.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import sys
import time

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(label: str, ok: bool, detail: str = "", hard: bool = True):
    mark = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"[{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        (FAILURES if hard else WARNINGS).append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-ok", action="store_true", help="GPU checks → warnings")
    ap.add_argument("--model", default=None, help="load tokenizer+config only")
    args = ap.parse_args()
    gpu_hard = not args.cpu_ok

    print("=== python ===")
    vi = sys.version_info
    check("python >= 3.12", vi >= (3, 12), f"{vi.major}.{vi.minor}.{vi.micro}")

    import torch

    print("=== torch / CUDA ===")
    check("torch == 2.14.0", torch.__version__.split("+")[0] == "2.14.0", torch.__version__)
    cuda = torch.cuda.is_available()
    check("cuda available", cuda, hard=gpu_hard)
    if cuda:
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        arch = torch.cuda.get_arch_list()
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"       device: {name}, CC {cc}, archs {arch}, VRAM {vram:.1f} GiB")
        check("compute capability == (6,1) (Pascal)", cc == (6, 1), f"got {cc}", hard=False)
        if cc == (6, 1):
            check(
                "sm_61 kernels in wheel",
                any("sm_61" in a or "sm_60" in a for a in arch),
                str(arch),
            )
        check("VRAM >= 10 GiB", vram >= 10, f"{vram:.1f} GiB")

        print("=== fp32 vs fp16 matmul (2048^2, 5 iters) ===")
        a = torch.randn(2048, 2048, device="cuda")
        for dt in (torch.float32, torch.float16):
            x = a.to(dt)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                _ = x @ x
            torch.cuda.synchronize()
            print(f"       {dt}: {(time.time() - t0) / 5 * 1000:.1f} ms/matmul")
    else:
        print("       (no CUDA — skipped device checks)")

    print("=== library pins ===")
    import datasets
    import peft
    import transformers
    import trl

    check("transformers == 5.17.0", transformers.__version__ == "5.17.0", transformers.__version__)
    check("trl == 1.13.0", trl.__version__ == "1.13.0", trl.__version__)
    check("peft == 0.20.0", peft.__version__ == "0.20.0", peft.__version__)
    check("datasets >= 5", int(datasets.__version__.split(".")[0]) >= 5, datasets.__version__)

    if args.model:
        print(f"=== model load ({args.model}, tokenizer+config only) ===")
        try:
            from transformers import AutoConfig, AutoTokenizer

            cfg = AutoConfig.from_pretrained(args.model)
            tok = AutoTokenizer.from_pretrained(args.model)
            check(
                "model config + tokenizer load",
                True,
                f"{cfg.model_type}, vocab {len(tok)}",
            )
        except Exception as e:  # noqa: BLE001
            check("model config + tokenizer load", False, repr(e))

    print()
    for w in WARNINGS:
        print(f"warning: {w}")
    if FAILURES:
        print(f"PREFLIGHT FAILED: {len(FAILURES)} hard failure(s)")
        return 1
    print("preflight OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
