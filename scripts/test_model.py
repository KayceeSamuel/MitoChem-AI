import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------- Config ---------------
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT = "point_to_checkpoint"  # set to checpoint directory
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Mito-Chem AI Inference Test")
print("=" * 60)

# ---------- Load Model and Tokenizer -----
# Load base Model
print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load tokenizer (from fine-tuned path to ensure correct special tokens)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load LoRA weights and merge
print("Loading and merging LoRA weights...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT, is_trainable=False)
model = model.merge_and_unload()
model.eval()

print("Model ready for inference")


# -------------helper inference -----------------
def ask(instruction, context="", max_tokens=256, temperature=0.7):
    """Ask fine-tuned biology model a question."""
    # Format the prompt using the chat template (User/Assistant)
    if context:
        prompt = f"""### User
Instruction:
{instruction}

Context:
{context}

### Assistant
"""
    else:
        prompt = f"""### User
Instruction:
{instruction}

### Assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(
        outputs[0][inputs["inputs_ids"].shape[1] :], skip_special_tokens=True
    )
    return response.strip()


# ----------run sample tests --------------
print("\n" + "=" * 60)
print("RUNNING SAMPLE TESTS")
print("=" * 60)

# The critical test for the MPC conflict (corrected definition, hopefully!)
instruction_mpc = "In the context of mitochondrial biology, what does MPC stand for and what is its primary function?"
result_mpc = ask(instruction=instruction_mpc, temperature=0.5, max_tokens=1000)
print(f" Q: {instruction_mpc}\n\nA:\n{result_mpc}\n")

# Test a complex pathway
instruction_pathway = "Explain the molecular mechanism by which MPC1 and MPC2 regulate mitochondrial pyruvate import and their role in cellular metabolism."
result_pathway = ask(instruction=instruction_pathway, temperature=0.5, max_tokens=1000)
print(f"\n Q: {instruction_pathway}\n\n A:\n{result_pathway}\n")
