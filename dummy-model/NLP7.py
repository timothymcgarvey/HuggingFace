from datasets import load_dataset

dataset = load_dataset(
    "mteb/amazon_reviews_multi",
    revision="refs/convert/parquet"
)

print(dataset)
def show_samples(dataset, num_samples=30, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Title: {example['id']}'")
        print(f"'>> Review: {example['text']}'")

english_dataset = dataset.filter(lambda x: str(x["id"]).startswith("en_"))
print(english_dataset)

english_dataset.set_format("pandas")
print(english_dataset)
english_df = english_dataset["train"][:]
show_samples(english_dataset)
# Show counts for top 20 products
