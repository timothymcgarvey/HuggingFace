
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]


from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state)
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits)

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
model.save_pretrained("Downloads")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)



#import transformers
#print(transformers.__version__)

from transformers import BertModel, BertTokenizer




#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#bert_model = BertModel.from_pretrained('bert-base-uncased')
#text_input="Company X reported higher-than-expected earnings..."
#inputs=tokenizer(text_input, return_tensors='pt',truncation=True,max_length=512)

#text_outputs=bert_model(**inputs)
#print(text_outputs)

#print("Loaded BERT successfully!")
#from transformers import BertModel, BertTokenizer


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from huggingface_hub import InferenceClient
import requests
import json

class OllamaInferenceClient(InferenceClient):
    def __init__(self, base_url="http://localhost:11434", model="llama3:latest"):
        self.base_url = base_url
        self.model = model

    def chat_completion(self, messages, temperature=0.8, max_tokens=200, top_p=0.95):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p
            }
        }

        # Tell requests to stream the response
        response = requests.post(url, json=payload, stream=True)

        # Collect streamed JSON lines
        content = ""
        for line in response.iter_lines():
            if line:
                try:
                    msg = json.loads(line)
                    if "message" in msg and "content" in msg["message"]:
                        content += msg["message"]["content"]
                except json.JSONDecodeError:
                    continue  # skip incomplete fragments

        # Return a Hugging Face–like object
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Msg", (), {"content": content})()
            })()]
        })

# --- Example usage ---
client = OllamaInferenceClient()
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story about a talking mountain."},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
