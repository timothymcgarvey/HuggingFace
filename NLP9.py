from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Replace with your actual repo name from the Hub
model_id = "tim11trade15machine/codeparrot-ds"

# Load tokenizer and model directly from the Hub
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="tim11trade15machine/codeparrot-ds", device=device
)

txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

txt = """\
# dataframe with profession, income and name
df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

# calculate the mean income per profession
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

import ast

def syntactic_validity(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

samples = [pipe(prompt, max_new_tokens=128)[0]["generated_text"] for prompt in prompts]
valid_rate = sum(syntactic_validity(s) for s in samples) / len(samples)
print(f"Syntactically valid completions: {valid_rate*100:.1f}%")
