# Kubernetes LoRA Fine-Tuning (Qwen2.5-1.5B)

A clean, minimal project for fine-tuning a small instruct model with LoRA/QLoRA on Kubernetes command/instruction data.

## What this project does
- Prepares supervised fine-tuning data from two Hugging Face datasets.
- Uses **ComponentSoft/k8s-kubectl-cot-20k** as the primary corpus.
- Uses **sozercan/k8s-instructions** as a small supplemental source for manifest-style examples.
- Trains a LoRA adapter (or QLoRA when 4-bit is available) with TRL `SFTTrainer`.
- Runs inference with the saved adapter.
- Compares base model responses vs fine-tuned responses on fixed Kubernetes prompts.

## Why LoRA / QLoRA
- LoRA reduces trainable parameters, making fine-tuning practical on a single GPU.
- QLoRA (4-bit loading + LoRA adapters) lowers VRAM usage further while preserving useful adaptation quality.
- This keeps training realistic for a single developer machine.

## Why the larger kubectl dataset is primary
- `ComponentSoft/k8s-kubectl-cot-20k` provides broader coverage for kubectl command behavior and metadata.
- The smaller `sozercan/k8s-instructions` dataset is useful for YAML/manifest examples but is intentionally sampled as supplemental data only.
- The project avoids using chain-of-thought fields as training targets.

## Folder structure
```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── prepare_data.py
├── train.py
├── infer.py
├── compare.py
├── data/
│   └── train.jsonl              # generated
└── outputs/                     # generated adapters/tokenizer
```

## Setup
### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
python -m pip install -r requirements.txt
```

## Data preparation
```bash
python prepare_data.py --output data/train.jsonl --supplemental-limit 300
```

What it does:
- Loads both datasets from Hugging Face.
- Uses `train` split when available (otherwise first available split with a warning).
- Builds `prompt`/`completion` pairs.
- Uses the main kubectl fields (`question`, `command`, `description`, `syntax`, `flags`) from the primary dataset.
- Avoids chain-of-thought text in targets.
- Prints sample transformed records.

## Training
### QLoRA (recommended when bitsandbytes 4-bit is available)
```bash
python train.py \
  --train-file data/train.jsonl \
  --output-dir outputs \
  --use-4bit \
  --num-train-epochs 1 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 8
```

### Standard LoRA fallback
```bash
python train.py \
  --train-file data/train.jsonl \
  --output-dir outputs \
  --num-train-epochs 1
```

Notes:
- Base model default is `Qwen/Qwen2.5-1.5B-Instruct`.
- Adapter and tokenizer are saved to `outputs/`.
- Script validates that `data/train.jsonl` includes `prompt` and `completion` columns.

## Inference with saved adapter
```bash
python infer.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path outputs \
  --prompt "How do I inspect a pod's logs in namespace prod?"
```

## Base vs fine-tuned comparison
```bash
python compare.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path outputs
```

The script evaluates fixed prompts covering:
- kubectl command explanation
- syntax/flag usage
- simple manifest generation

## Expected outputs
- `prepare_data.py`: creates `data/train.jsonl` and prints sample records.
- `train.py`: creates LoRA adapter files and tokenizer files in `outputs/`.
- `infer.py`: prints one prompt and one model answer.
- `compare.py`: prints side-by-side outputs for base and fine-tuned models.

## Troubleshooting
- **Out of memory**: reduce `--max-length`, lower batch size, or increase gradient accumulation.
- **bitsandbytes unavailable**: run without `--use-4bit` for standard LoRA.
- **Slow downloads**: model and dataset pulls from Hugging Face can take time; retry after network stabilization.
- **Adapter load errors**: verify training completed and `outputs/` contains adapter config/weights.
- **Dataset schema changes**: rerun `prepare_data.py` and inspect printed sample records to confirm mapping.

## Resume-ready project description
Built a lightweight Kubernetes domain fine-tuning pipeline using Qwen2.5-1.5B-Instruct + LoRA/QLoRA (Transformers, PEFT, TRL, Datasets, Accelerate), with custom data preparation from large-scale kubectl instruction data and side-by-side base-vs-adapted model evaluation scripts.
