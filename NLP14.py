import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "huggingface-course/codeparrot-ds"

# ------------------------------
# Device selection (MPS-aware)
# ------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ------------------------------
# Load model + tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
model.eval()

# ------------------------------
# Test prompts (add as many as you like)
# ------------------------------
TEST_PROMPTS = [
    # --- Code tasks ---




    """import numpy as np

def average(arr):
""",

]

# ------------------------------
# Inference function
# ------------------------------
def generate(prompt, max_new_tokens=60, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,       # for creative completions
            pad_token_id=tokenizer.eos_token_id
        )

    elapsed = time.time() - start

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = outputs[0].shape[-1] - inputs["input_ids"].shape[-1]

    return text, tokens_generated, elapsed

# ------------------------------
# Systematic test runner
# ------------------------------
def run_tests():
    print("\n============================")
    print(" MODEL INFERENCE TEST SUITE")
    print("============================\n")

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n---- Test {i+1} ----")
        print("Prompt:")
        print(prompt)
        print("------------------")

        text, tokens, t = generate(prompt)

        print("Generated:")
        print(text)
        print(f"\nTokens generated: {tokens}")
        print(f"Time: {t:.3f}s ({tokens/t:.2f} tokens/sec)")
        print("------------------")

if __name__ == "__main__":
    run_tests()
