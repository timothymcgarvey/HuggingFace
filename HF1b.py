import requests
from datasets import load_dataset
import json
from collections import defaultdict
import re

# Load your previously saved comment file
with open("all_comments.json", "r") as f:
    comments = json.load(f)

print(f"Loaded {len(comments)} total comments")

issue_comments = defaultdict(list)

for c in comments:
    issue_url = c.get("issue_url", "")
    match = re.search(r"/issues/(\d+)", issue_url)
    if match:
        issue_num = int(match.group(1))
        issue_comments[issue_num].append(c.get("body", ""))

print(f"Built comment lookup for {len(issue_comments)} issues")
issues_dataset = load_dataset("json", data_files="datasets-with-comments.jsonl")
issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": issue_comments.get(int(x["number"]), []) if x["number"] else []}
)


sample = issues_with_comments_dataset["train"].shuffle(seed=42).select(range(5))

for n, c in zip(sample["number"], sample["comments"]):
    print(f"Issue #{n}: {len(c)} comments")
    for comment in c[:2]:  # show first two comments
        print("  →", comment[:120].replace("\n", " "), "...")

train_ds = issues_with_comments_dataset["train"]
train_ds.to_json("datasets-with-comments.jsonl")
train_ds.save_to_disk("datasets_with_comments")
