from transformers import AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
#encoding = tokenizer(example)
#print(type(encoding))
#print(encoding.tokens())
from transformers import pipeline

token_classifier = pipeline("token-classification",aggregation_strategy="simple")
print(token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn."))