"""Train a Kubernetes domain LoRA/QLoRA adapter with TRL SFTTrainer."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_bnb_config(use_4bit: bool) -> BitsAndBytesConfig | None:
    if not use_4bit:
        return None
    if not torch.cuda.is_available():
        print("[WARN] 4-bit requested but CUDA is not available. Falling back to standard LoRA.")
        return None

    try:
        import bitsandbytes  # noqa: F401

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[WARN] 4-bit requested but unavailable ({exc}). Falling back to standard LoRA.")
        return None


def resolve_torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def render_chat_example(tokenizer: AutoTokenizer, prompt: str, completion: str) -> str:
    user_content = prompt.replace("System: ", "").replace("User: ", "").replace("Assistant:", "").strip()
    assistant_content = completion.strip()
    messages = [
        {"role": "system", "content": "You are a Kubernetes and AI infrastructure assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return f"{prompt}\n{completion}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--use-4bit", action="store_true", help="Enable QLoRA 4-bit loading.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = Path(args.train_file)
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_file}. Run `python prepare_data.py --output {train_file}` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = resolve_bnb_config(args.use_4bit)
    torch_dtype = resolve_torch_dtype()

    model_kwargs = {
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        if torch.cuda.is_available():
            # Explicit single-GPU placement for stable 4-bit runs.
            model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if torch.cuda.is_available() and bnb_config is None:
        try:
            model = model.to("cuda")
        except Exception as exc:
            raise RuntimeError(
                "Failed to move model to CUDA. Check CUDA-enabled PyTorch/bitsandbytes install."
            ) from exc

    train_ds = load_dataset("json", data_files=str(train_file), split="train")
    required_columns = {"prompt", "completion"}
    missing = required_columns.difference(set(train_ds.column_names))
    if missing:
        raise ValueError(
            f"Training file is missing required columns: {sorted(missing)}. "
            f"Found columns: {train_ds.column_names}"
        )
    train_ds = train_ds.map(
        lambda ex: {"text": render_chat_example(tokenizer, ex["prompt"], ex["completion"])},
        remove_columns=train_ds.column_names,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

    sft_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "report_to": "none",
        "dataset_text_field": "text",
        "dataloader_pin_memory": torch.cuda.is_available(),
    }
    sft_signature = inspect.signature(SFTConfig.__init__).parameters
    if "max_seq_length" in sft_signature:
        sft_kwargs["max_seq_length"] = args.max_length
    elif "max_length" in sft_signature:
        sft_kwargs["max_length"] = args.max_length
    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_ds,
        "peft_config": peft_config,
    }
    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved adapter and tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()
