#creating a GPT2 tokenizer with the BPE model from scratch
from datasets import load_dataset
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")



def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
tokenizer = Tokenizer(models.BPE())
#BPE tokenizers don't use normalization so we go right to pre-tokenization
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!"))

#training
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
start, end = encoding.offsets[4]
print(sentence[start:end])

tokenizer.decoder = decoders.ByteLevel()

print(tokenizer.decode(encoding.ids))

from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)

tokenizer.save("GPT2tokenizerBPE.json")


