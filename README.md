# Finetune Pipeline Updated

This repository contains a 3-stage fine-tuning pipeline to train a Small Language Model (Gemma-4 E2B-it) for car repair queries using QLoRA.

## Files Overview
- `1_pre_eval.py`: Evaluates the base model before fine-tuning.
- `2_finetune.py`: Fine-tunes the model on the dataset.
- `3_post_eval.py`: Evaluates the fine-tuned model and compares it with the baseline.

## Changes: Old Code vs New Code

### 1. Gemma-4 System Prompt
**Old Code:**
```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": question},
]
```
**New Code:**
```python
messages = [
    {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"},
]
```
**Reason:** Gemma-4 does not natively support the `system` role. Merging it ensures the training format exactly matches the inference format.

### 2. Training Message Role
**Old Code:**
```python
{"role": "assistant", "content": answer}
```
**New Code:**
```python
{"role": "model", "content": answer}
```
**Reason:** Gemma-4 expects the response role to be `model` instead of `assistant`.

### 3. Training Config (Epochs, LR, Alpha)
**Old Code:**
```python
epochs: int = 10
learning_rate: float = 2e-4
lora_alpha: int = 32
```
**New Code:**
```python
epochs: int = 3
learning_rate: float = 1e-4
lora_alpha: int = 16
```
**Reason:** 10 epochs and a 2e-4 learning rate on a small 342-row dataset caused severe overfitting. Reduced to 3 epochs, 1e-4 LR, and alpha=rank(16) for stable domain adaptation.

### 4. Max Sequence Length
**Old Code:**
```python
max_seq_length: int = 512
```
**New Code:**
```python
max_seq_length: int = 1024
```
**Reason:** Matches the 1024 length used in the evaluation scripts (Stage 1 & 3) so training and evaluation lengths are aligned.

### 5. Checkpoint Saving & Regularization
**Old Code:**
```python
# No checkpoint saving or weight decay
```
**New Code:**
```python
load_best_model_at_end=True,
weight_decay=0.01,
```
**Reason:** The final epoch is often overfitted. This saves the model with the lowest validation loss and adds L2 regularization to prevent overfitting.

### 6. Judge Model Update
**Old Code:**
```python
eval_model: str = "groq/llama-3.1-8b-instant"
```
**New Code:**
```python
eval_model: str = "groq/gpt-oss-20b"
```
**Reason:** The old model is being deprecated by Groq. The new 20B model provides better evaluation reasoning.
