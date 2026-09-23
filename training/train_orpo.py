"""ORPO fine-tuning to make the agent's answers friendlier (project bonus).

ORPO (Odds Ratio Preference Optimization) combines SFT and preference alignment
in one pass — no separate reward model and no reference model in memory, which
is what makes it fit on a single free-tier GPU.

Needs CUDA. Run on Colab (T4 is enough for Qwen2.5-3B in 4-bit):

    python training/train_orpo.py --pairs data/preference_data.json

Afterwards, serve the merged model through Ollama and point the agent at it:

    LLM_BACKEND=ollama OLLAMA_MODEL=qwen-almaty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
DEFAULT_PAIRS = Path(__file__).parent.parent / "data" / "preference_data.json"


def load_pairs(path: Path):
    from datasets import Dataset

    pairs = json.loads(path.read_text(encoding="utf-8"))
    if not pairs:
        raise RuntimeError(f"{path} is empty — run training/build_preference_data.py first")
    print(f"loaded {len(pairs)} preference pairs")
    return Dataset.from_list(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="outputs/orpo-almaty")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.1, help="ORPO lambda / odds-ratio weight")
    args = parser.parse_args()

    import torch
    from unsloth import FastLanguageModel

    # TRL moved ORPO under trl.experimental in recent releases; fall back to the
    # top-level import so this works on either version.
    try:
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer
    except ImportError:
        from trl import ORPOConfig, ORPOTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — ORPO training needs a GPU runtime")

    dataset = load_pairs(args.pairs)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        # Recompute activations instead of storing them. ORPO forwards both the
        # chosen and rejected sequence, so activation memory is roughly doubled
        # versus plain SFT — on a 14 GB T4 that is the difference between
        # training and an OOM.
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainer = ORPOTrainer(
        model=model,
        # `tokenizer=` was renamed to `processing_class=` across TRL trainers.
        processing_class=tokenizer,
        train_dataset=dataset,
        args=ORPOConfig(
            output_dir=args.out,
            # Effective batch stays 8; the split is 1x8 rather than 2x4 because
            # prompts now carry the afisha context and are several hundred
            # tokens longer than they used to be.
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=8e-6,   # preference tuning wants a much lower LR than SFT
            beta=args.beta,
            # ORPOConfig has no max_prompt_length — max_length covers
            # prompt + completion together.
            max_length=1024,
            logging_steps=10,
            # Left at its default, this forks worker processes (num_proc = CPU
            # count, 2 on a Colab T4 box) to tokenize the dataset — but by this
            # point the 4-bit model is already loaded and CUDA is already
            # initialized in the parent process. Forking after that can leave
            # a duplicated/orphaned CUDA context in the child, which silently
            # eats several GB of VRAM the trainer itself never allocated and
            # was the actual cause of the "OOM on a 352 MiB alloc while GPU is
            # basically empty" crash, not the training step itself.
            dataset_num_proc=1,
            optim="adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",
            seed=42,
        ),
    )

    print("training ORPO — watch that loss decreases and rewards/margins grows")
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
