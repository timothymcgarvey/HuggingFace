from datasets import load_dataset
from huggingface_hub import HfApi

raw_datasets = load_dataset("opus_books", "en-fr")


split_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=20)
split_datasets["validation"] = split_datasets.pop("test")
print(split_datasets)

#print(split_datasets["train"][1]["translation"])

from transformers import pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"

translator=pipeline("translation",model=model_checkpoint)

print(translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format I sent that to you via email."
))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,return_tensors="pt")
en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)
#input_ids for the the English sentence and labels for the French sentence
#wrong_targets = tokenizer(fr_sentence)
#print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
#print(tokenizer.convert_ids_to_tokens(inputs["labels"]))

max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

#finetuning the model.  we use Seq2SeqTrainer this time

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

#data collation with a collator that can deal with labels (for the french text) and inputIDs (for the English text)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])

import evaluate

metric = evaluate.load("sacrebleu")

predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
print(metric.compute(predictions=predictions, references=references))

import numpy as np


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


#finetuning the model



from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,
    bf16=True,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print(trainer.evaluate(max_length=max_length))

trainer.train()

trainer.push_to_hub(tags="translation", commit_message="Training complete")










