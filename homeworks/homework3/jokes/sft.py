import os
import gc
import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig

torch.cuda.set_per_process_memory_fraction(0.85, device=0)

@dataclass
class SFTArgs:
    output_dir: str = "qwen3-0.6b-anecdote"
    max_length: int = 256
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 4
    learning_rate: float = 6e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    logging_steps: int = 30
    save_steps: int = 500
    eval_strategy: str = "no"
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 5
    max_steps: int = -1
    
@dataclass
class LoraArgs:
    r: int = 16
    lora_alpha: int = 32
    target_modules: list = ("q_proj", "k_proj", "v_proj", "o_proj")
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

def train(
    model_id: str = "Qwen/Qwen3-0.6B",
    *,
    dataset_path: str | None = None,
    sft_args: SFTArgs = SFTArgs(),
    lora_args: LoraArgs = LoraArgs(),    
):
    # model_id = "Qwen/Qwen3-0.6B"
    data = load_dataset("json", data_files=dataset_path or "anecd.jsonl", split="train[:2000]")

    bnb = BitsAndBytesConfig(load_in_8bit=True)

    print("loading model...")
    max_mem = {0: "7GiB", "cpu": "30GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        max_memory=max_mem,
        low_cpu_mem_usage=True,
        dtype=torch.float16,
    )

    cfg = SFTConfig(**vars(sft_args))

    model = prepare_model_for_kbit_training(model)
    
    peft_cfg = LoraConfig(**vars(lora_args), base_model_name_or_path=model_id)
    trainer_kwargs = {
        "model": model,
        "train_dataset": data,
        "formatting_func": None,
        "args": cfg,
    }

    trainer_kwargs["peft_config"] = peft_cfg

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(cfg.output_dir)
    
if __name__ == "__main__":
    train()
