"""Run inference with a base model plus fine-tuned LoRA adapter."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT = "How do I list all pods across namespaces and show wide output?"


def build_prompt(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a Kubernetes and AI infrastructure assistant."},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (
            "You are a Kubernetes and AI infrastructure assistant. "
            "Provide concise and practical answers.\n"
            f"User: {user_prompt}\n"
            "Assistant:"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default="outputs")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (
        torch.float16 if use_cuda else torch.float32
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    base_model = base_model.to(device)
    try:
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load adapter from '{args.adapter_path}'. "
            "Run training first or verify the adapter path."
        ) from exc
    model.eval()

    prompt = build_prompt(tokenizer, args.prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = response.split("Assistant:")[-1].strip()

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
