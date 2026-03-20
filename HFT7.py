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

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]")) #WordPiece tokenizer

tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

#tokenizer.normalizer = normalizers.Sequence(
 #   [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
#)

print(tokenizer.normalizer.normalize_str(u"\u0085"))

#tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer() #splits on punctuation and whitespace
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() #splits on punctuation and whitespace
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

pre_tokenizer = pre_tokenizers.WhitespaceSplit() #only splits on whitespace
print(pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
) #we're using 'Sequence to conjoin several pre-tokenizers
print(pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

#now we need to train the WordPiece model that we already specified.  We need to add the special tokens that we want as well
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer) #we can train the model using the iterator that we defined earlier
#tokenizer.model = models.WordPiece(unk_token="[UNK]")
#tokenizer.train(["wikitext-2.txt"], trainer=trainer)
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
#we need to know the ids of the special tokens
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)

tokenizer.decoder = decoders.WordPiece(prefix="##")
print(tokenizer.decode(encoding.ids))

tokenizer.save("tokenizer.json")

new_tokenizer = Tokenizer.from_file("tokenizer.json")

from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)



