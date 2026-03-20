from datasets import load_dataset

# Option A: Python subset via Nan-Do
raw_datasets = load_dataset("Nan-Do/code-search-net-python")

print(raw_datasets["train"])
print(raw_datasets["train"][123456]["original_string"])

training_corpus = (
    raw_datasets["train"][i : i + 1000]["original_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)

gen = (i for i in range(10))
print(list(gen))
print(list(gen))
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["original_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )


training_corpus = get_training_corpus()
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
print(tokens)
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
tokens = tokenizer.tokenize(example)
print(tokens)
tokenizer.save_pretrained("code-search-net-tokenizer")
tokenizer.push_to_hub("code-search-net-tokenizer")