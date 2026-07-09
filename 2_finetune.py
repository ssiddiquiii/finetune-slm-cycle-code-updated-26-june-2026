# ============================================================
# CELL 1 — DEPENDENCY INSTALLATION & KAGGLE ENVIRONMENT SETUP
# ● SETS PYTORCH CUDA ALLOC CONFIG TO PREVENT VRAM FRAGMENTATION
# ● PINS TRL, DATASETS, PROTOBUF TO STABLE COMPATIBLE VERSIONS
# ● INSTALLS LATEST UNSLOTH BUILD WITH KAGGLE-SPECIFIC PATCHES
# ● DEPLOYS PEFT, ACCELERATE, BITSANDBYTES FOR QLORA TRAINING
# ● LOCKS ALL BINARY DEPENDENCIES BEFORE FRAMEWORK INITIALIZATION
# ============================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Installing stable training stack...")
!pip install -q --no-warn-conflicts "trl>=0.18.2,<=0.24.0" "datasets>=3.4.1,<4.4.0" "protobuf>=3.20.3,<6.0.0"
!pip install -q --no-warn-conflicts -U unsloth
!pip install -q --no-warn-conflicts -U "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-warn-conflicts peft accelerate bitsandbytes huggingface_hub sentencepiece deepeval litellm

print("\n" + "="*60)
print("✅ INSTALLATIONS COMPLETE")
print("="*60)

# ============================================================
# CELL 2 — FRAMEWORK VERSION CHECK & CUDA HARDWARE TELEMETRY
# ● SILENCES UNSLOTH MULTI-MODAL PLACEHOLDER WARNINGS ON STARTUP
# ● IMPORTS UNSLOTH FIRST TO APPLY TRITON KERNEL PATCHES
# ● VERIFIES ALL LIBRARY VERSIONS: TORCH, TRANSFORMERS, PEFT, TRL
# ● SCANS ACTIVE GPU DEVICES WITH VRAM CAPACITY REPORTING
# ● CONFIRMS TRAINING ENVIRONMENT IS READY FOR QLORA FINE-TUNING
# ============================================================

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import warnings
warnings.filterwarnings("ignore") # Brutally ignore ALL Python warnings (not just FutureWarning)

# SILENCE EARLY FRAMEWORK CHATTER: Suppresses multi-modal placeholder warning hooks on startup
logging.getLogger("unsloth").setLevel(logging.ERROR)
logging.getLogger("unsloth_zoo").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import unsloth  # CRITICAL: Must be first among deep learning libraries to apply Triton patches
import torch
import transformers
import peft
import trl
import accelerate
import bitsandbytes
import sys

print("=" * 60)
print("KAGGLE ENVIRONMENT CHECK")
print("=" * 60)
print(f"Python:        {sys.version.split()[0]}")
print(f"Torch:         {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"Unsloth:       {unsloth.__version__}")

print("\nCUDA STATUS")
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {name} ({vram:.1f} GB)")
else:
    print("\n✗ GPU NOT detected")

print("\n✅ ENGINE READY")

# ============================================================
# CELL 3 — SECRETS RETRIEVAL & DUAL API AUTHENTICATION
# ● READS HF_TOKEN AND GROQ_API_KEY FROM KAGGLE SECRETS VAULT
# ● BINDS ALL CREDENTIALS INTO RUNTIME OS ENVIRONMENT VARIABLES
# ● SUPPRESSES LITELLM DEBUG NOISE AND TOKENIZER PARALLELISM LOGS
# ● AUTHENTICATES HUGGINGFACE HUB SESSION FOR GATED MODEL ACCESS
# ● CONFIRMS GROQ API KEY IS AVAILABLE FOR OPTIONAL JUDGE USE
# ============================================================

import os
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
GROQ_API_KEY = user_secrets.get_secret("GROQ_API_KEY")

os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HUGGINGFACE_TOKEN'] = HF_TOKEN
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['LITELLM_SUPPRESS_DEBUG_INFO'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

login(token=HF_TOKEN, add_to_git_credential=False)
print("\n✓ APIs authenticated")

# ============================================================
# CELL 4 — QLORA TRAINING CONFIGURATION & DIRECTORY SETUP
# ● DEFINES BASE MODEL ID AND HF DATASET REPOSITORY TARGETS
# ● SETS LORA RANK (R=8) AND ALPHA (16) FOR DOMAIN ADAPTATION
# ● CONFIGURES TRAINING: 1 EPOCH, LR=2E-5, MAX_SEQ=1024
# ● CALIBRATES BATCH SIZE AND GRADIENT ACCUMULATION FOR T4 VRAM
# ● CREATES CHECKPOINT AND ADAPTER OUTPUT DIRECTORIES
# ============================================================

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    model_id: str = "unsloth/gemma-4-E2B-it"
    dataset_repo: str = "ssiddiquii/car-repair-hq-342" 
    
    lora_r: int = 8                   # ANTI-FLUFF FIX: 16→8. Restricts capacity to memorize fluff.
    lora_alpha: int = 16              # ANTI-FLUFF FIX: alpha=2*rank is a standard to stabilize training.
    lora_dropout: float = 0           # Unsloth optimized path requires dropout=0.
    
    epochs: int = 3                   # BASELINE: 3 epochs
    per_device_batch_size: int = 4    # Tuned for Gemma-4 on T4: balances throughput vs VRAM.
    grad_accumulation: int = 4        # Effective batch = 4 × 4 = 16. Mirrors original 1×16 setup.
    learning_rate: float = 1e-4       # BASELINE: 1e-4 learning rate
    max_seq_length: int = 1024        # FIX: 512→1024. Must match eval scripts to avoid unfair scoring.
    warmup_steps: float = 0.03
    
    work_dir: str = "/kaggle/working"

CONFIG = Config()
WORK = Path(CONFIG.work_dir)
CHECKPOINTS = WORK / "checkpoints_qlora"
ADAPTERS = WORK / "lora_adapters_qlora"
for d in [CHECKPOINTS, ADAPTERS]:
    d.mkdir(parents=True, exist_ok=True)

print("\n✓ QLORA CONFIG LOCKED")

# ============================================================
# CELL 5 — TRAIN & VALIDATION SPLIT STREAMING FROM HF HUB
# ● STREAMS TRAIN AND VALIDATION SPLITS FROM HF PARQUET STORE
# ● LOADS NATIVELY WITHOUT CONVERTING DATASET INTO MEMORY LISTS
# ● REPORTS ROW COUNTS FOR BOTH TRAIN AND VALIDATION SPLITS
# ● CONFIRMS DATASET CONNECTIVITY BEFORE EXPENSIVE MODEL LOAD
# ● USES HF_TOKEN FOR GATED PRIVATE DATASET ACCESS
# ============================================================

from datasets import load_dataset

print(f"Loading natively cached Parquet slices from {CONFIG.dataset_repo}...")

train_ds_raw = load_dataset(CONFIG.dataset_repo, split="train", token=HF_TOKEN)
val_ds_raw = load_dataset(CONFIG.dataset_repo, split="validation", token=HF_TOKEN)

print(f"✓ Train rows fetched: {train_ds_raw.num_rows}")
print(f"✓ Validation rows fetched: {val_ds_raw.num_rows}")

# ============================================================
# CELL 6 — DATA FORMATTING WITH GEMMA-4 CHAT TEMPLATE
# ● MERGES SYSTEM PROMPT INTO USER TURN FOR GEMMA-4 COMPATIBILITY
# ● MAPS QUESTION/ANSWER PAIRS TO USER/MODEL MESSAGE ROLES
# ● ENSURES TRAIN FORMAT IS IDENTICAL TO INFERENCE FORMAT
# ● PROCESSES DATASETS NATIVELY WITHOUT MEMORY LIST CONVERSION
# ● SHUFFLES TRAINING SPLIT WITH SEED=42 FOR REPRODUCIBILITY
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert car repair assistant. Answer the user's question concisely "
    "and accurately. Be technically precise about parts, diagnostics, and procedures."
)

def to_messages_structured(item):
    # GEMMA-4 FORMAT FIX: Gemma-4 has no native system role.
    # System prompt merged into user turn — identical to inference format in 1_pre_eval.py and 3_post_eval.py.
    # Using 'model' role (not 'assistant') — Gemma-4's native response role token.
    # CRITICAL: Train format MUST be 100% identical to infer format. Even 1 token diff = degraded results.
    return {
        "messages": [
            {"role": "user",  "content": f"{SYSTEM_PROMPT.strip()}\n\n{str(item['question']).strip()}"},
            {"role": "model", "content": str(item['answer']).strip()},
        ]
    }

# Process datasets natively without breaking into memory lists
train_ds = train_ds_raw.map(to_messages_structured, remove_columns=train_ds_raw.column_names)
val_ds = val_ds_raw.map(to_messages_structured, remove_columns=val_ds_raw.column_names)

# Natively shuffle the training array
train_ds = train_ds.shuffle(seed=42)

print(f"\n✓ Aligned multi-turn training data mapped. Sample schema: {train_ds[0]}")

# ============================================================
# CELL 7 — GEMMA-4 MODEL LOAD, QLORA INJECTION & TOKENIZATION
# ● CLEARS VRAM RINGS BEFORE MODEL WEIGHT ALLOCATION
# ● LOADS GEMMA-4 E2B-IT VIA UNSLOTH FASTMODEL (FP16 + 4-BIT)
# ● INJECTS QLORA ADAPTERS ONTO ALL ATTENTION AND MLP LAYERS
# ● ENABLES UNSLOTH GRADIENT CHECKPOINTING FOR VRAM EFFICIENCY
# ● PRE-RENDERS DATASET TO FLAT TEXT FOR SFTTRAINER CONSUMPTION
# ============================================================

import gc
import torch
from unsloth import FastModel

# SILENCE AUDIO/VISION HOOK LOG SPAM (already set at top, but reaffirm after model imports)
logging.getLogger("unsloth").setLevel(logging.ERROR)

# Clear VRAM rings before model memory allocation
gc.collect()
torch.cuda.empty_cache()

print("\nLoading Gemma-4 E2B-it via Unsloth in 4-BIT QLoRA...")

model, tokenizer = FastModel.from_pretrained(
    model_name=CONFIG.model_id,
    max_seq_length=CONFIG.max_seq_length,
    load_in_4bit=True,                  
    dtype=None,                          # GEMMA-4 OOM FIX: Let Unsloth auto-select dtype to prevent massive float32 VRAM spikes.
    full_finetuning=False,
    token=HF_TOKEN,
)

# EXPLICIT STATE LOCK: Pre-emptively disable caching to silence runtime trainer logs
model.config.use_cache = False

model = FastModel.get_peft_model(
    model,
    r=CONFIG.lora_r,
    lora_alpha=CONFIG.lora_alpha,
    lora_dropout=CONFIG.lora_dropout,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # REVERT: We MUST use gradient checkpointing to prevent OOM on 16GB T4.
    use_gradient_checkpointing=True,
    random_state=42,
    use_rslora=False,
)

print("✓ LoRA adapters injected. Gradient checkpointing enabled (to prevent OOM).")

# ============================================================
# THE ELEGANT ZERO-MEMORY FIX: GEMMA-4 DTYPE MISMATCH
# Unsloth forces float32 training, so intermediate activations are float32.
# But Gemma-4's per_layer projection and gate weights remain float16.
# Casting all weights to fp32 caused a CUDA OOM (16GB T4 isn't enough).
# FIX: Dynamically cast the fp32 activation to fp16 just-in-time 
# using a universal pre-forward hook strictly on all "per_layer" modules.
# ============================================================
def _fix_per_layer_dtype(module, args):
    x = args[0]
    # BUG FIX: Only cast floating-point tensors (activations). Never cast integer indices (input_ids).
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x) and hasattr(module, "weight"):
        if x.dtype != module.weight.dtype:
            return (x.to(module.weight.dtype),) + args[1:]
    return args

_hook_count = 0
for name, module in model.named_modules():
    # BUG FIX: Strictly limit to Linear layers. Prevents hooking Embedding layers like `embed_tokens_per_layer`.
    if "per_layer" in name and isinstance(module, torch.nn.Linear):
        module.register_forward_pre_hook(_fix_per_layer_dtype)
        _hook_count += 1

print(f"✓ DTYPE FIX: Registered JIT dtype cast hooks on {_hook_count} Gemma-4 per-layer modules.")

def render_row(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

train_ds = train_ds.map(render_row, remove_columns=["messages"], desc="Rendering train")
val_ds = val_ds.map(render_row, remove_columns=["messages"], desc="Rendering val")

print(f"\n✓ Datasets ready for SFTTrainer")

# ============================================================
# CELL 8 — QLORA SUPERVISED FINE-TUNING WITH VALIDATION LOOP
# ● DETECTS GEMMA-4 CHAT TEMPLATE BOUNDARIES DYNAMICALLY
# ● TRAINS ONLY ON RESPONSE TOKENS (IGNORES PROMPT TOKEN LOSS)
# ● EVALUATES ON VALIDATION SET EVERY 10 STEPS FOR MONITORING
# ● SAVES BEST CHECKPOINT BY EVAL LOSS (NOT FINAL EPOCH WEIGHTS)
# ● USES PAGED ADAMW 8-BIT OPTIMIZER FOR T4 MEMORY EFFICIENCY
# ============================================================

import math
import warnings
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# Suppress the specific unsloth batch warning if any residual logs remain
warnings.filterwarnings("ignore", message=".*num_items_in_batch.*")

# Dynamic Template Boundary Detection
sample_text = train_ds[0]["text"]
if "<|turn>user" in sample_text:
    INSTRUCTION_PART, RESPONSE_PART = "<|turn>user\n", "<|turn>model\n"
elif "<start_of_turn>user" in sample_text:
    INSTRUCTION_PART, RESPONSE_PART = "<start_of_turn>user\n", "<start_of_turn>model\n"


# BUG FIX: These values are now sourced from CONFIG to ensure Config class actually controls training.
total_steps = math.ceil(
    len(train_ds) * CONFIG.epochs / (CONFIG.per_device_batch_size * CONFIG.grad_accumulation)
)
computed_warmup_steps = math.ceil(total_steps * CONFIG.warmup_steps)

print(f"--- ARCHITECTURAL OPTIMIZATION METRICS ---")
print(f"  Total Dataset Rows:       {len(train_ds)}")
print(f"  Total Optimization Steps: {total_steps}")
print(f"  Enforced Warmup Steps:    {computed_warmup_steps}\n")

sft_config = SFTConfig(
    output_dir=str(CHECKPOINTS),
    num_train_epochs=CONFIG.epochs,
    
    # SYNCED FROM CONFIG: Both values now sourced from Config class — changing Config actually works.
    per_device_train_batch_size=CONFIG.per_device_batch_size,
    gradient_accumulation_steps=CONFIG.grad_accumulation,
    
    learning_rate=CONFIG.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=computed_warmup_steps,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit", 
    bf16=False,                   # GEMMA-4 FIX: Gemma-4 requires float32 training. fp16/bf16 crash on per_layer gates.
    fp16=False,                   # GEMMA-4 FIX: Unsloth confirms: "float16 precision for gemma4 won't work"
    max_seq_length=CONFIG.max_seq_length,
    dataset_text_field="text", 
    logging_steps=5,
    weight_decay=0.05,            # ANTI-FLUFF FIX: 0.01→0.05. Stronger L2 regularization against overfitting.
    
    eval_strategy="steps",
    eval_steps=5,         # CRITICAL FIX: 10→5. With 16 total steps, this gives 3 eval points.
    save_strategy="steps",
    save_steps=5,         # CRITICAL FIX: 20→5. save_steps=20 with 16 total steps = ZERO checkpoints saved!
    save_total_limit=3,           # Keep 3 checkpoints to ensure best is not evicted.
    load_best_model_at_end=True,  # CRITICAL: loads the best val-loss checkpoint, not the final one.
    metric_for_best_model="eval_loss",  # Lower loss = better model.
    greater_is_better=False,      # For loss, lower is better.
    
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model, 
    args=sft_config, 
    train_dataset=train_ds, 
    eval_dataset=val_ds, 
    tokenizer=tokenizer
)
trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

print("Starting QLoRA training loop...")
from unsloth import unsloth_train 
train_result = unsloth_train(trainer) 

print(f"\n✓ TRAINING COMPLETE | Final loss: {train_result.training_loss:.4f}")

# ============================================================
# CELL 9 — LORA ADAPTER SAVE & ZIP FOR KAGGLE DATASET EXPORT
# ● SAVES FINE-TUNED LORA ADAPTER WEIGHTS TO WORKING DIRECTORY
# ● SAVES TOKENIZER ALONGSIDE ADAPTERS FOR COMPLETE ARTIFACT
# ● CREATES ZIP ARCHIVE FOR KAGGLE OUTPUT PANEL DOWNLOAD
# ● PREPARES ADAPTER PACKAGE AS INPUT FOR STAGE 3 POST-EVAL
# ● CONFIRMS ARTIFACT IS READY FOR HUGGINGFACE HUB UPLOAD
# ============================================================

import shutil

print(f"\nSaving adapters to {ADAPTERS}...")
model.save_pretrained(str(ADAPTERS))
tokenizer.save_pretrained(str(ADAPTERS))

zip_path = f"{ADAPTERS}.zip"
shutil.make_archive(base_name=str(ADAPTERS), format="zip", root_dir=str(ADAPTERS))

print(f"✓ Zip created: {zip_path}")
print("Ready for Post-Eval fusion processing.")
