"""Compare base model outputs against LoRA-adapted outputs on fixed Kubernetes prompts."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_PROMPTS = [
    "Explain this command and when to use it: kubectl get pods -A -o wide",
    "What does --namespace do in kubectl commands? Give one example.",
    "Generate a minimal Kubernetes Deployment manifest for an nginx app with 2 replicas.",
]


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
            "Be concise, practical, and technically accurate.\n"
            f"User: {user_prompt}\n"
            "Assistant:"
        )


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 220) -> str:
    full_prompt = build_prompt(tokenizer, prompt)
    target_device = model.get_input_embeddings().weight.device
    inputs = tokenizer(full_prompt, return_tensors="pt").to(target_device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default="outputs")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    base_model.eval()

    tuned_base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    try:
        tuned_model = PeftModel.from_pretrained(tuned_base, args.adapter_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load adapter from '{args.adapter_path}'. "
            "Run training first or verify the adapter path."
        ) from exc
    tuned_model.eval()

    for i, prompt in enumerate(TEST_PROMPTS, start=1):
        print("\n" + "=" * 100)
        print(f"Prompt {i}: {prompt}")
        print("-" * 100)

        base_output = generate(base_model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        tuned_output = generate(tuned_model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)

        print("[Base Model]")
        print(base_output)
        print("\n[Fine-tuned Adapter]")
        print(tuned_output)


if __name__ == "__main__":
    main()
