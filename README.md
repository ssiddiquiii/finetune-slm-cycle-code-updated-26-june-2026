# Car Specialist SLM — Fine-Tune Pipeline

A 3-stage fine-tuning pipeline to build an offline Car Specialist AI using **Google Gemma-4 E2B-it (2B)** with QLoRA on Kaggle T4 GPU.

The goal is to train a lightweight model that can answer car repair questions (parts, diagnostics, procedures) and run fully offline on mobile/edge devices.

---

## What Each File Does

| File | Stage | Purpose |
|------|-------|---------|
| `1_pre_eval.py` | Stage 1 | Evaluate the base model BEFORE fine-tuning (get a baseline score) |
| `2_finetune.py` | Stage 2 | Fine-tune Gemma-4 on the car repair dataset using QLoRA |
| `3_post_eval.py` | Stage 3 | Evaluate the fine-tuned model and compare scores vs baseline |

---

## Tech Stack

- **Model:** Gemma-4 E2B-it via Unsloth (4-bit QLoRA)
- **Dataset:** `ssiddiquii/car-repair-hq-342` on HuggingFace
- **Training:** Kaggle T4 GPU (Free Tier)
- **Evaluation:** DeepEval with `groq/gpt-oss-20b` as judge

---

## How to Run

Run each file as a **separate Kaggle notebook** in order:

1. Run `1_pre_eval.py` → download the output zip → upload as Kaggle Dataset
2. Run `2_finetune.py` → download adapters zip → upload as Kaggle Dataset
3. Run `3_post_eval.py` (mount both datasets) → see the delta report

Add `HF_TOKEN` and `GROQ_API_KEY` to Kaggle Secrets before running.

---

## Changes from Old Code

### Bug Fixes

**1. Gemma-4 system prompt fix (all 3 files)**
- Old code injected a `{"role": "system"}` message which Gemma-4 does not support natively
- This caused a mismatch between training format and inference format
- Fixed by merging the system prompt directly into the user message

**2. Training role changed from `assistant` to `model`**
- Gemma-4 uses `model` as the response role, not `assistant`
- This ensures the training data matches Gemma's actual token structure

**3. Typo fix in `3_post_eval.py`**
- `from deeval.test_case` → `from deepeval.test_case` (missing letter `p`)
- Was silently failing and falling back to wrong namespace

### Training Improvements

**4. Epochs reduced from 10 to 3**
- 342 training rows with 10 epochs = guaranteed overfitting
- The model was memorizing writing style instead of learning car repair knowledge
- 3 epochs gives better generalization

**5. Learning rate changed from `2e-4` to `1e-4`**
- `2e-4` is too aggressive for a small 342-row dataset
- `1e-4` is the standard QLoRA recommendation for domain fine-tuning

**6. LoRA alpha changed from `32` to `16`**
- `alpha = rank` ratio is more stable for small datasets
- Higher alpha was causing the LoRA updates to be too aggressive

**7. Max sequence length fixed from `512` to `1024`**
- Stage 1 and Stage 3 both use 1024, but training was using 512
- This mismatch made the eval unfair for longer questions

**8. Best checkpoint saving added**
- Added `load_best_model_at_end=True` to SFTConfig
- Old code always saved the final epoch weights which are usually overfit
- Now saves the checkpoint with the lowest validation loss instead

**9. Weight decay added (`0.01`)**
- Basic L2 regularization to help prevent overfitting on small datasets

### Other Updates

**10. Judge model updated**
- Changed from `groq/llama-3.1-8b-instant` to `groq/gpt-oss-20b`
- Groq deprecated `llama-3.1-8b-instant` on June 26, 2026 (off August 16, 2026)
- `gpt-oss-20b` is Groq's official replacement with better reasoning

**11. Cell headers updated in all 3 files**
- Each cell now has a proper name and bullet points describing what it does
- Makes the notebooks easier to read and understand

---

## Training Config

```python
model_id      = "unsloth/gemma-4-E2B-it"
epochs        = 3
learning_rate = 1e-4
lora_r        = 16
lora_alpha    = 16
max_seq_length = 1024
weight_decay  = 0.01
```

---

## Required Kaggle Secrets

- `HF_TOKEN` — for HuggingFace model and dataset access
- `GROQ_API_KEY` — for the judge model (evaluation)
