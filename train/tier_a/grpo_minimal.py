"""A1: hand-written GRPO loop (no TRL) for the doc->todos task.

Pure-PyTorch GRPO: sample G completions per prompt, group-standardize rewards,
clipped surrogate on per-token ratios, k3 KL to the frozen (adapter-disabled)
reference. Meant to run on a 1080 Ti (CC 6.1, torch 2.14.0+cu126) or CPU.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from train.tier_a.common import (
    LocalSpec,
    RunLogger,
    batch_indices,
    completion_mask,
    evaluate,
    load_config,
    pick_device,
    seq_logprobs,
)

EPS_ADV = 1e-6


def grpo_advantages(rewards: list[float], G: int) -> torch.Tensor:
    """Group-standardize rewards: A_i = (r_i - mean_g)/(std_g + 1e-6)."""
    r = torch.tensor(rewards, dtype=torch.float32)
    out = torch.empty_like(r)
    for s in range(0, len(rewards), G):
        g = r[s : s + G]
        out[s : s + G] = (g - g.mean()) / (g.std(unbiased=False) + EPS_ADV)
    return out


def grpo_loss_terms(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    adv: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token clipped surrogate + k3 KL. Returns (loss, surrogate, kl_mean).

    logp_* are [B, T] per-token logprobs over completion positions;
    adv is [B] broadcast over tokens; mask is [B, T] (1 on real completion tokens).
    """
    ratio = torch.exp(logp_new - logp_old)
    a = adv.unsqueeze(1)
    sur_tok = torch.minimum(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
    surrogate = (sur_tok * mask).sum() / mask.sum().clamp(min=1)
    x = logp_ref - logp_new
    kl_tok = torch.exp(x) - x - 1
    kl = (kl_tok * mask).sum() / mask.sum().clamp(min=1)
    loss = -surrogate + beta * kl
    return loss, surrogate, kl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier_a_grpo.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = pick_device(args.device)
    gcfg, lcfg = cfg["grpo"], cfg["lora"]
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    from datasets import disable_progress_bar
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, cfg["dtype"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["alpha"],
            lora_dropout=lcfg["dropout"],
            target_modules=lcfg["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    if gcfg.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=gcfg["lr"])

    spec = LocalSpec.from_config(cfg)
    total_reward, schema_fn = spec.reward_funcs
    logger = RunLogger(cfg["out_dir"])
    out_dir = Path(cfg["out_dir"])

    def run_eval(tag: str):
        metrics = evaluate(
            model,
            tokenizer,
            spec.eval_dataset,
            spec.scorer,
            max_new_tokens=gcfg["max_completion_length"],
            device=device,
            eval_batch_size=cfg["data"].get("eval_batch_size", 8),
        )
        (out_dir / f"eval_{tag}.json").write_text(json.dumps(metrics, indent=2))
        print(f"[eval {tag}] {metrics}")
        return metrics

    eval_before = run_eval("before")

    n_rows = len(spec.train_dataset)
    B, G = gcfg["batch_prompts"], gcfg["num_generations"]

    for step in range(1, gcfg["steps"] + 1):
        t0 = time.time()
        rows = [spec.train_dataset[i] for i in batch_indices(rng, n_rows, B)]
        texts = [
            tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
            for r in rows
        ]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gcfg["max_prompt_length"],
        ).to(device)
        prompt_len = enc["input_ids"].shape[1]

        was_training = model.training
        model.eval()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=gcfg["temperature"],
                num_return_sequences=G,
                max_new_tokens=gcfg["max_completion_length"],
                pad_token_id=tokenizer.pad_token_id,
            )
        if was_training:
            model.train()
        completions = gen[:, prompt_len:]
        texts_out = tokenizer.batch_decode(completions, skip_special_tokens=True)

        rewards = total_reward(
            completions=texts_out,
            doc=[r["doc"] for r in rows for _ in range(G)],
            gold_todos=[r["gold_todos"] for r in rows for _ in range(G)],
            source_entities=[r["source_entities"] for r in rows for _ in range(G)],
        )
        schemas = schema_fn(
            completions=texts_out,
            doc=[r["doc"] for r in rows for _ in range(G)],
            gold_todos=[r["gold_todos"] for r in rows for _ in range(G)],
            source_entities=[r["source_entities"] for r in rows for _ in range(G)],
        )
        adv = grpo_advantages(rewards, G).to(device)
        completion_part = completion_mask(gen, prompt_len, tokenizer.eos_token_id)
        prompt_part = enc["attention_mask"].repeat_interleave(G, dim=0)
        attention_mask = torch.cat((prompt_part, completion_part), dim=1)
        mask = completion_part

        logp_new = seq_logprobs(model, gen, attention_mask, prompt_len)
        logp_old = logp_new.detach()
        with torch.no_grad(), model.disable_adapter():
            logp_ref = seq_logprobs(model, gen, attention_mask, prompt_len)

        loss, _surrogate, kl = grpo_loss_terms(
            logp_new, logp_old, logp_ref, adv, mask, gcfg["epsilon"], gcfg["beta"]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()

        r_t = torch.tensor(rewards)
        comp_lens = mask.sum(1)
        # per-sequence mean log-ratio (x = logp_ref - logp_new) for the viz batch
        x_tok = (logp_ref - logp_new.detach()) * mask
        seq_x = (x_tok.sum(1) / comp_lens.clamp(min=1)).tolist()
        # first prompt's group only: the viz batch must be one prompt's G samples
        logger.set_sample_batch(
            comp_lens[:G].tolist(), schemas[:G], seq_x[:G], rewards[:G]
        )
        logger.write_samples(
            step,
            [
                {
                    "prompt_id": rows[i // G]["task_id"],
                    "completion": texts_out[i],
                    "reward": rewards[i],
                }
                for i in range(min(4, len(rewards)))
            ],
        )
        logger.log(
            step,
            r_t.mean().item(),
            r_t.std(unbiased=False).item() if len(rewards) > 1 else 0.0,
            kl.item(),
            comp_lens.mean().item(),
            sum(schemas) / len(schemas),
        )
        print(
            f"step {step}/{gcfg['steps']} reward={r_t.mean().item():.3f} "
            f"kl={kl.item():.4f} loss={loss.item():.4f} len={comp_lens.mean().item():.0f} "
            f"schema={sum(schemas) / len(schemas):.2f} {time.time() - t0:.1f}s/step",
            flush=True,
        )

        if step % cfg["save_every"] == 0:
            model.save_pretrained(out_dir / f"ckpt-{step}")

    model.save_pretrained(out_dir / "ckpt-final")
    eval_after = run_eval("after")
    logger.set_eval(eval_before, eval_after)
    logger.close()


if __name__ == "__main__":
    main()
