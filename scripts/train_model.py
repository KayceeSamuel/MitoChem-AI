import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# --------- PATHS & CONFIGURATION --------------
BASE_MODEL = "Qwen/Qwen2.5-7B_Instruct"
DATA_PATH = "../data/training_data_mixed.jsonl"
OUTPUT_DIR = "checkpoints/mitochem_v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("=" * 60)
print("Mito-Chem AI Training Setup")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 60)

# ----------Load Dataset --------------------
print("\n Loading dataset ....")
# Load dataset
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")


def format_example(ex):
    """Formats the data into the User/Assistant conversation template."""
    instr = ex.get("instruction") or ex.get("prompt") or ""
    # build context block
    context_parts = []
    if "context" in ex and ex["context"]:
        context_parts.append(f"Context:\n{ex['context']}")
    if "smiles" in ex and ex["smiles"]:
        context_parts.append(f"SMILES: {ex['smiles']}")
    if "cell_summary" in ex and ex["cell_summary"]:
        context_parts.append(f"Data:\n{ex['cell_summary']}")

    context_block = "\n\n".join(context_parts)

    if context_block:
        full_prompt = f"Instruction:\n{instr}\n\n{context_block}"
    else:
        full_prompt = f"Instruction:\n{instr}"

    # Get response, handling non-string JSON outputs
    resp = ex.get("response", "")
    if not isinstance(resp, str):
        import json

        resp = json.dumps(resp, ensure_ascii=False)
    # Final formatted text for SFTTrainer
    ex["text"] = f"### User\n{full_prompt}\n\n### Assistant\n{resp}"
    return ex


ds = raw_ds.map(format_example, remove_columns=raw_ds.column_names)
print("Dataset loaded: {len(ds)} examples")

# ---------- Load Model with Quantization -----------
print("\n Loading base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.use_cache = False

print(f"Model loaded")

# ------------ Setup LoRA ---------------
print("\n Setting up LoRA...")
lora_config = LoraConfig(
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
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# --------- Safer training arguments -----------------
print("\n Configuring training")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # training schedule
    num_train_epochs=1,  # no of epochs
    per_device_train_batch_size=4,  # batch size
    gradient_accumulation_steps=2,  # accumulation
    # learning rate
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    # gradient safety
    max_grad_norm=0.3,
    # logging and saving
    logging_steps=25,
    save_steps=250,
    save_total_limit=3,
    # Optimization
    bf16=True,
    optim="paged_adamw_8bit",
    # Other
    report_to="none",
    gradient_checkpointing=True,  # save memory
)
print(" Training configured")
print(f" Learning rate: {training_args.learning_rate}")
print(f" Max grad norm: {training_args.max_grad_norm}")
print(
    f" Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
)

# ---------- Tokenize dataset ---------------
print("\n Tokenizing dataset ...")
MAX_LEN = 2048


def tokenize_fn(ex):
    out = tokenizer(
        ex["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    out["labels"] = out["input_ids"].copy()
    return out


tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_ds = tokenized_ds.shuffle(seed=42)

print(f" Dataset tokenized: {len(tokenized_ds)} examples")

# ----------- create trainer ----------------
print("\n Creating trainer ...")

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=tokenized_ds,
    args=training_args,
)
print("Trainer ready")

# ------------ Train ----------------------
print("\n" + "=" * 60)
print("Starting training")
print("\n" + "=" * 60)

# start training
trainer.train()

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)

# ------------- Save final model -----------
print("\nSaving final model...")

trainer.save_model(OUTPUT_DIR + "/final")
tokenizer.save_pretrained(OUTPUT_DIR + "/final")

print(f" Model saved to: {OUTPUT_DIR}/final")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
