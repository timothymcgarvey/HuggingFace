
from datasets import load_dataset, DownloadConfig

data_files = "https://huggingface.co/datasets/casinca/PUBMED_title_abstracts_2019_baseline/resolve/main/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset(
    "json",
    data_files=data_files,
    split="train",
    download_config=DownloadConfig(delete_extracted=True),  # optional argument
)

print(pubmed_dataset)
print(pubmed_dataset[0])

import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

print(f"Dataset size in bytes: {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")

import timeit

code_snippet = """batch_size = 1000

for idx in range(0, len(pubmed_dataset), batch_size):
    _ = pubmed_dataset[idx:idx + batch_size]
"""

time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
print(
    f"Iterated over {len(pubmed_dataset)} examples (about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb/time:.3f} GB/s"
)

pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)

next(iter(pubmed_dataset_streamed))

dataset_head = pubmed_dataset_streamed.take(5)
print(list(dataset_head))




from datasets import load_dataset

ds = load_dataset("datajuicer/the-pile-freelaw-refined-by-data-juicer",
                  split="train", streaming=True)
sample = next(iter(ds))
print(sample["text"][:500])
from itertools import islice
from datasets import interleave_datasets
from datasets import interleave_datasets

# Normalize so both have 'text' and 'meta'
def normalize_with_meta(example):
    # Some datasets have 'meta' as JSON or dict; we normalize it to string
    meta = example.get("meta", None)
    if meta is None:
        meta = {}
    elif not isinstance(meta, str):
        try:
            import json
            meta = json.dumps(meta)
        except Exception:
            meta = str(meta)
    return {
        "text": example.get("text", ""),
        "meta": meta
    }

pubmed_norm = pubmed_dataset_streamed.map(normalize_with_meta)
freelaw_norm = ds.map(normalize_with_meta)

# Interleave safely (they now share the same schema)
combined_dataset = interleave_datasets([pubmed_norm, freelaw_norm])

print(list(islice(combined_dataset, 2)))
