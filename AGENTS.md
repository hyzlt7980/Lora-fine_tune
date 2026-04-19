# AGENTS.md

Repository guidance for contributors and coding agents.

## Principles
- Keep implementations simple, readable, and easy to run locally.
- Prefer straightforward Python scripts over frameworks or heavy abstractions.
- Use minimal dependencies required for data prep, training, and inference.
- Avoid over-engineering and premature optimization.

## Coding style
- Write clear, explicit code with descriptive variable names.
- Add concise comments only when they improve understanding.
- Favor one clean implementation path unless multiple are truly needed.
- Keep function boundaries practical and easy to modify.

## Project expectations
- This repo focuses on LoRA/QLoRA fine-tuning for a Kubernetes domain assistant.
- Prioritize runnable scripts (`prepare_data.py`, `train.py`, `infer.py`, `compare.py`).
- Keep command-line examples in `README.md` copy-paste runnable.
- Whenever commands, defaults, or paths change, update `README.md` in the same change.

## Data and training
- Use `ComponentSoft/k8s-kubectl-cot-20k` as the primary training source.
- Use `sozercan/k8s-instructions` only as a small supplemental source.
- Do not train on chain-of-thought fields.
- Save processed SFT data to `data/train.jsonl`.

## Output hygiene
- Save training artifacts to `outputs/`.
- Keep generated or large artifacts out of version control via `.gitignore`.
