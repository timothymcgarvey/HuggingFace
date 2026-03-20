
def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False

filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
example_1 = "import numpy as np"
example_2 = "import pandas as pd"

print(
    any_keyword_in_string(example_1, filters), any_keyword_in_string(example_2, filters)
)

from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset


def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)


# This cell will take a very long time to execute, so you should skip it and go to
# the next one!
from datasets import load_dataset

split = "train"  # "valid"
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

#data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
#filtered_data = filter_streaming_dataset(data, filters)

from datasets import load_dataset, DatasetDict

ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)

print(raw_datasets)

for key in raw_datasets["train"][0]:
    print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

    from transformers import AutoTokenizer

    context_length = 128
    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

    outputs = tokenizer(
        raw_datasets["train"][:2]["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    #getting rid of the end chunks that are less than 128 characters

def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Replace with your actual repo name from the Hub
model_id = "tim11trade15machine/codeparrot-ds"

# Load tokenizer and model directly from the Hub
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="tim11trade15machine/codeparrot-ds", device=device
)
keytoken_ids = []
for keyword in [
    "plt",
    "pd",
    "sk",
    "fit",
    "predict",
    " plt",
    " pd",
    " sk",
    " fit",
    " predict",
    "testtest",
]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        print(f"Keyword has not single token: {keyword}")

from torch.nn import CrossEntropyLoss
import torch



def keytoken_weighted_loss(inputs, logits, keytoken_ids_tensor, alpha=1.0):

    # SHIFT for LM training: predict token n+1 from token n
    shift_labels = inputs[..., 1:].contiguous()          # [batch, S-1]
    shift_logits = logits[..., :-1, :].contiguous()      # [batch, S-1, V]

    # Per-token cross-entropy
    loss_fct = CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)
    ).view(shift_logits.size(0), shift_logits.size(1))    # [batch, S-1]

    # Keyword mask on ORIGINAL tokens
    # inputs: [batch, S]
    # keytoken_ids_tensor: [K]
    mask = (inputs[..., None] == keytoken_ids_tensor).any(-1).float()  # [batch, S]

    # Align mask to shifted length (S -> S-1)
    mask = mask[:, :token_loss.size(1)]                  # now [batch, S-1]

    # Weight tokens where any keyword matched
    token_weights = alpha * (1.0 + mask)                 # [batch, S-1]

    # Weighted mean
    return (token_loss * token_weights).mean()




from torch.utils.data.dataloader import DataLoader

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=64, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=64)

weight_decay = 0.1


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
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
    return mean_loss.item(), float(torch.exp(mean_loss))


from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
context_length = 128
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    n_layer=3,
    n_head=3,
    n_embd=192,
)

model = GPT2LMHeadModel(config)


try:
    model.transformer = torch.compile(model.transformer)
except:
    pass


from torch.optim import AdamW

optimizer = AdamW(get_grouped_params(model), lr=5e-4)

from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="fp16")
# Build tensor ONCE — not inside the loss function
keytoken_ids_tensor = torch.tensor(
    keytoken_ids,
    device=accelerator.device,
    dtype=torch.long,
)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)

from huggingface_hub import Repository, get_full_repo_name

model_name = "codeparrot-ds-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)

output_dir = "codeparrot-ds-accelerate"
#repo = Repository(output_dir, clone_from=repo_name)

print(evaluate())

from tqdm import tqdm

gradient_accumulation_steps = 1
eval_steps = 5_000

model.train()
completed_steps = 0
samples_per_step=32

model.train()
completed_steps = 0

for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1),
        total=len(train_dataloader)
    ):
        logits = model(batch["input_ids"]).logits

        # Use tensor version of keytoken IDs
        loss = keytoken_weighted_loss(
            batch["input_ids"],
            logits,
            keytoken_ids_tensor,
            alpha=1.0,
        )

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1

        if step % 100 == 0:
            accelerator.print(f"Step {step}  Loss {loss.item():.4f}")
        import os
        import torch
        from transformers import GPT2LMHeadModel

        checkpoint_dir = output_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        if (step % eval_steps) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                # ---------------------------------------------------------
                # 1. Get wrapped weights from Accelerate
                # ---------------------------------------------------------
                raw_state_dict = model.state_dict()

                # ---------------------------------------------------------
                # 2. Remove “transformer._orig_mod.” prefix
                # ---------------------------------------------------------
                cleaned_state_dict = {}
                for key, value in raw_state_dict.items():
                    new_key = key.replace("transformer._orig_mod.", "transformer.")
                    cleaned_state_dict[new_key] = value

                # ---------------------------------------------------------
                # 3. Load into a clean CPU model
                # ---------------------------------------------------------
                cpu_model = GPT2LMHeadModel(config)
                cpu_model.load_state_dict(cleaned_state_dict, strict=True)

                # ---------------------------------------------------------
                # 4. Save & push to Hub
                # ---------------------------------------------------------
                cpu_model.save_pretrained(
                    checkpoint_dir,
                    push_to_hub=True,
                    safe_serialization=True,
                    commit_message=f"Training step {step}",
                )

                tokenizer.save_pretrained(checkpoint_dir, push_to_hub=True)



