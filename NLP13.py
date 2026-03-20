import os
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoConfig,
    GPT2LMHeadModel,
    get_scheduler,
)

from accelerate import Accelerator
from huggingface_hub import HfApi, create_repo


# =========================
# CONFIG
# =========================

@dataclass
class Config:
    # Hugging Face repo where you want to push
    repo_id: str = "tim11trade15machine/codeparrot-ds-accelerate"

    # Model + tokenizer
    base_model_name: str = "gpt2"
    tokenizer_name: str = "huggingface-course/code-search-net-tokenizer"

    # Model size (tiny GPT-2)
    n_layer: int = 3
    n_head: int = 3
    n_embd: int = 192

    # Training
    context_length: int = 128
    batch_size: int = 64
    num_train_epochs: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1_000

    # Logging / eval / checkpointing
    logging_steps: int = 100
    eval_steps: int = 5_000  # in optimizer steps (global steps)
    output_dir: str = "codeparrot-ds-accelerate"

    # Keyword weighting
    alpha: float = 1.0
    keywords = [
        "plt", "pd", "sk", "fit", "predict",
        " plt", " pd", " sk", " fit", " predict",
        "testtest",
    ]


cfg = Config()


# =========================
# HELPERS
# =========================

def upload_folder_retry(repo_id, folder_path, commit_msg, retries=5, delay=5):
    """Robust upload to HF Hub with retries."""
    api = HfApi()
    for attempt in range(1, retries + 1):
        try:
            api.upload_folder(
                repo_id=repo_id,
                folder_path=folder_path,
                commit_message=commit_msg,
                ignore_patterns=["*.lock"],
            )
            print(f"[HF] Upload succeeded on attempt {attempt}")
            return
        except Exception as e:
            print(f"[HF] Upload attempt {attempt} failed: {e}")
            if attempt == retries:
                print("[HF] Giving up after max retries.")
                raise
            time.sleep(delay)


def get_grouped_params(model, weight_decay=0.1, no_decay=("bias", "LayerNorm.weight")):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def clean_state_dict(state_dict):
    """Remove the '._orig_mod.' prefixes introduced by Accelerate wrapping."""
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("transformer._orig_mod.", "transformer.")
        cleaned[new_key] = v
    return cleaned


# =========================
# KEYWORD WEIGHTED LOSS
# =========================

def keytoken_weighted_loss(inputs, logits, keytoken_ids_tensor, alpha=1.0):
    """
    inputs: [batch, seq_len]
    logits: [batch, seq_len, vocab_size]
    keytoken_ids_tensor: [K]  (tensor of token ids to emphasize)
    """
    # LM shift: predict token n+1 from token n
    shift_labels = inputs[..., 1:].contiguous()        # [B, S-1]
    shift_logits = logits[..., :-1, :].contiguous()    # [B, S-1, V]

    # Per-token CrossEntropy
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_logits.size(0), shift_logits.size(1))  # [B, S-1]

    # Keyword mask on original tokens
    # inputs: [B, S]
    # keytoken_ids_tensor: [K]
    mask = (inputs[..., None] == keytoken_ids_tensor).any(-1).float()  # [B, S]

    # Align mask to shifted length
    if mask.size(1) != token_loss.size(1):
        mask = mask[:, :token_loss.size(1)]  # [B, S-1] after crop

    token_weights = alpha * (1.0 + mask)  # [B, S-1]

    weighted_loss = (token_loss * token_weights).mean()
    return weighted_loss


# =========================
# MAIN
# =========================

def main():
    accelerator = Accelerator(mixed_precision="fp16")  # good on MPS
    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    # -------- Tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------- Datasets --------
    raw_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    raw_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

    raw_datasets = DatasetDict(
        {
            "train": raw_train,
            "valid": raw_valid,
        }
    )

    def tokenize_fn(examples):
        outputs = tokenizer(
            examples["content"],
            truncation=True,
            max_length=	cfg.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == cfg.context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["valid"],
        batch_size=cfg.batch_size,
    )

    # -------- Model --------
    config = AutoConfig.from_pretrained(
        cfg.base_model_name,
        vocab_size=len(tokenizer),
        n_ctx=cfg.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
    )

    model = GPT2LMHeadModel(config)

    optimizer = AdamW(
        get_grouped_params(model, weight_decay=cfg.weight_decay),
        lr=cfg.learning_rate,
    )

    # Prepare with Accelerate
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # -------- Key tokens --------
    keytoken_ids = []
    for kw in cfg.keywords:
        ids = tokenizer(kw, add_special_tokens=False).input_ids
        if len(ids) == 1:
            keytoken_ids.append(ids[0])
        else:
            accelerator.print(f"[KW] '{kw}' is not a single token, skipping.")
    if not keytoken_ids:
        accelerator.print("[KW] No valid single-token keywords found, loss will be unweighted.")
    keytoken_ids_tensor = torch.tensor(keytoken_ids, device=device, dtype=torch.long) if keytoken_ids else torch.tensor([], device=device, dtype=torch.long)

    # -------- Scheduler --------
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = cfg.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # -------- HF Repo Setup (once) --------
    if accelerator.is_main_process:
        create_repo(cfg.repo_id, exist_ok=True)
        os.makedirs(cfg.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # -------- Eval function --------
    def evaluate():
        model.eval()
        losses = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(batch["input_ids"], labels=batch["input_ids"])
                loss = outputs.loss
            loss = accelerator.gather(loss)
            losses.append(loss)
        model.train()
        mean_loss = torch.mean(torch.stack(losses))
        try:
            ppl = torch.exp(mean_loss).item()
        except OverflowError:
            ppl = float("inf")
        return mean_loss.item(), ppl

    # -------- Training Loop --------
    global_step = 0
    progress_bar = tqdm(
        range(num_training_steps),
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(cfg.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(batch["input_ids"])
            logits = outputs.logits

            if keytoken_ids_tensor.numel() > 0:
                loss = keytoken_weighted_loss(
                    batch["input_ids"],
                    logits,
                    keytoken_ids_tensor,
                    alpha=cfg.alpha,
                )
            else:
                # fallback: standard LM loss if no keywords
                shift_labels = batch["input_ids"][..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            if global_step % cfg.logging_steps == 0:
                accelerator.print(
                    f"Epoch {epoch+1} Step {global_step} - loss: {loss.item():.4f}"
                )

            if global_step % cfg.eval_steps == 0 or global_step == num_training_steps:
                eval_loss, ppl = evaluate()
                accelerator.print(
                    f"[EVAL] step {global_step}: loss={eval_loss:.4f}, ppl={ppl:.2f}"
                )

                # -------- Save + Push to Hub (robust) --------
                if accelerator.is_main_process:
                    # 1. Get raw (possibly wrapped) state_dict
                    raw_sd = model.state_dict()
                    cleaned_sd = clean_state_dict(raw_sd)

                    # 2. Build clean CPU model and load weights
                    cpu_model = GPT2LMHeadModel(config)
                    cpu_model.load_state_dict(cleaned_sd, strict=True)
                    cpu_model.save_pretrained(cfg.output_dir, safe_serialization=True)
                    tokenizer.save_pretrained(cfg.output_dir)

                    # 3. Upload folder with retries
                    upload_folder_retry(
                        repo_id=cfg.repo_id,
                        folder_path=cfg.output_dir,
                        commit_msg=f"Training step {global_step}",
                        retries=5,
                        delay=5,
                    )

        accelerator.print(f"Finished epoch {epoch+1}/{cfg.num_train_epochs}")

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
