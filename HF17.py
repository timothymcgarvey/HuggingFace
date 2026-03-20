from datasets import load_dataset
# Load JSONL file into a Hugging Face Dataset
remote_dataset = load_dataset(
    "json",
    data_files="datasets-with-comments.jsonl",
    split="train"
)
remote_dataset_ipr = remote_dataset.map(
    lambda x: {"is_pull_request": x["pull_request"] in [None, "None"]}
)
sample = remote_dataset_ipr.shuffle(seed=666).select(range(300))
for url, ipr in zip(sample["comments"], sample["is_pull_request"]):
    print(f">> comments: {url}")
    print(f">> Is pull request: {ipr}\n")
issues_dataset = remote_dataset_ipr.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
print(issues_dataset)

columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
print(issues_dataset)
issues_dataset.set_format("pandas")
df = issues_dataset[:]
print(df["comments"][1020].tolist())

comments_df = df.explode("comments", ignore_index=True)
print(comments_df.head(49))

from datasets import Dataset

comments_dataset = Dataset.from_pandas(comments_df)
print(comments_dataset)

comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)

comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
print(comments_dataset)

def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)

from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

import torch

device = torch.device("mps")
model.to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

embedding = get_embeddings(comments_dataset["text"][0])
print(embedding.shape)

embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

embeddings_dataset.add_faiss_index(column="embeddings")

question = "Why do I get FileNotFound error?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
print(question_embedding.shape)


print(embeddings_dataset.shape)

scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
import pandas as pd

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()
