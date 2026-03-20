
import time
import math
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

import pandas as pd

# Load the JSONL file produced by fetch_issues()
df = pd.read_json("datasets-issues.jsonl", lines=True)

# Coerce all columns to strings (Arrow-safe)
df = df.astype(str)

# Save back to a clean JSONL file
df.to_json("datasets-issues-clean.jsonl", orient="records", lines=True)

print("✅ Cleaned JSONL written to datasets-issues-clean.jsonl")

# Now safely load into Hugging Face Datasets
from datasets import load_dataset

issues_dataset = load_dataset("json", data_files="datasets-issues-clean.jsonl", split="train")
print(issues_dataset)
sample = issues_dataset.shuffle(seed=666).select(range(3))

# Print out the URL and pull request entries
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")