"""A2: GRPO via TRL's GRPOTrainer for the doc->todos task.

Same data/rewards as A1 (train.tier_a.grpo_minimal) through LocalSpec.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer, TrainerCallback

from train.tier_a.common import (
    LocalSpec,
    RunLogger,
    evaluate,
    load_config,
    pick_device,
    sample_batch_from_model,
)


class RunLoggerCallback(TrainerCallback):
    """Map TRL's logged metrics into RunLogger.log()."""

    def __init__(self, logger: RunLogger):
        self.logger = logger
        self._latest: dict = {}

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        self._latest.update(logs)

        def find(*keys, default=0.0):
            # exact match first, then substring (TRL 1.13 logs reward,
            # reward_std, kl, completions/mean_length, rewards/<fn>/mean)
            for want in keys:
                if want in logs:
                    return logs[want]
            for want in keys:
                for k, v in logs.items():
                    if want in k:
                        return v
            return default

        step = int(logs.get("step", state.global_step))
        if "reward" not in logs:
            return
        self.logger.log(
            step,
            reward_mean=find("reward"),
            reward_std=find("reward_std"),
            kl=find("kl"),
            completion_len=find("completions/mean_length", "mean_length"),
            schema_ok=find("rewards/schema_ok/mean", "schema_ok"),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier_a_grpo.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = pick_device(args.device)
    gcfg, lcfg = cfg["grpo"], cfg["lora"]
    out_dir = Path(cfg["out_dir_trl"])

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    spec = LocalSpec.from_config(cfg)
    logger = RunLogger(out_dir)

    grpo_args = GRPOConfig(
        output_dir=str(out_dir),
        max_steps=gcfg["steps"],
        per_device_train_batch_size=gcfg["batch_prompts"] * gcfg["num_generations"],
        num_generations=gcfg["num_generations"],
        max_completion_length=gcfg["max_completion_length"],
        temperature=gcfg["temperature"],
        epsilon=gcfg["epsilon"],
        beta=gcfg["beta"],
        loss_type="grpo",
        learning_rate=gcfg["lr"],
        use_vllm=False,
        use_cpu=(device == "cpu"),
        bf16=False,
        fp16=False,
        gradient_checkpointing=gcfg.get("grad_checkpointing", False),
        logging_steps=cfg["log_every"],
        save_steps=cfg["save_every"],
        seed=cfg["seed"],
        report_to=[],
        reward_weights=spec.reward_weights,
        model_init_kwargs={"dtype": cfg["dtype"]},
    )

    trainer = GRPOTrainer(
        model=cfg["model"],
        args=grpo_args,
        train_dataset=spec.train_dataset,
        reward_funcs=spec.reward_funcs,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["alpha"],
            lora_dropout=lcfg["dropout"],
            target_modules=lcfg["target_modules"],
            task_type="CAUSAL_LM",
        ),
        callbacks=[RunLoggerCallback(logger)],
    )

    before = evaluate(
        trainer.model,
        tokenizer,
        spec.eval_dataset,
        spec.scorer,
        max_new_tokens=gcfg["max_completion_length"],
        device=device,
        eval_batch_size=cfg["data"].get("eval_batch_size", 8),
    )
    (out_dir / "eval_before.json").write_text(json.dumps(before, indent=2))
    print(f"[eval before] {before}")

    trainer.train()
    trainer.save_model(str(out_dir / "ckpt-final"))

    metrics = evaluate(
        trainer.model,
        tokenizer,
        spec.eval_dataset,
        spec.scorer,
        max_new_tokens=gcfg["max_completion_length"],
        device=device,
        eval_batch_size=cfg["data"].get("eval_batch_size", 8),
    )
    (out_dir / "eval_after.json").write_text(json.dumps(metrics, indent=2))
    print(f"[eval after] {metrics}")

    # real sample batch: one train prompt group through the trained model
    row = spec.train_dataset[0]
    batch = sample_batch_from_model(
        trainer.model,
        tokenizer,
        row,
        spec.scorer,
        spec.reward_funcs[0],
        gcfg,
        device,
    )
    logger.set_sample_batch(**batch)
    logger.close()


if __name__ == "__main__":
    main()
