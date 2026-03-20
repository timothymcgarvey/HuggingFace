import ast
import torch
from transformers import pipeline

# ---------------------------
# 1. Setup
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_id = "codeparrot/codeparrot-small"
finetuned_model_id = "tim11trade15machine/codeparrot-ds"

pipe_base = pipeline("text-generation", model=base_model_id, device=device)
pipe_finetuned = pipeline("text-generation", model=finetuned_model_id, device=device)

# ---------------------------
# 2. Prompts to test
# ---------------------------
prompts = [
    "# create some data\nx = np.random.randn(100)\ny = np.random.randn(100)\n# create dataframe from x and y\n",
    "# dataframe with profession, income and name\nx, y, z = [], [], []\n# create dataframe\n",
    "# plot data\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\n# create plot\n",
    "# create dictionary from two lists\na = [1,2,3]\nb = ['a','b','c']\n# dictionary\n",
]

# ---------------------------
# 3. Utility functions
# ---------------------------
def syntactic_validity(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def runtime_validity(code):
    """Sandboxed runtime test — restrict builtins for safety."""
    try:
        exec(code, {"__builtins__": {"print": print, "range": range}})
        return True
    except Exception:
        return False

def evaluate_model(pipe, name):
    syntactic_ok = 0
    runtime_ok = 0
    lengths = []
    completions = []

    for prompt in prompts:
        result = pipe(
            prompt,
            max_new_tokens=128,
            temperature=0.3,
            top_p=0.9,
            num_return_sequences=1,
        )[0]["generated_text"]

        completions.append(result)
        lengths.append(len(result.split()))

        if syntactic_validity(result):
            syntactic_ok += 1
        if runtime_validity(result):
            runtime_ok += 1

    n = len(prompts)
    print(f"\n===== Results for {name} =====")
    print(f"Syntactic validity: {syntactic_ok}/{n} ({100*syntactic_ok/n:.1f}%)")
    print(f"Runtime success:   {runtime_ok}/{n} ({100*runtime_ok/n:.1f}%)")
    print(f"Average length:    {sum(lengths)/n:.1f} words")
    print("\nSample generations:\n")
    for p, c in zip(prompts, completions):
        print("Prompt:\n", p)
        print("Completion:\n", c)
        print("-" * 60)

# ---------------------------
# 4. Run comparisons
# ---------------------------
print(evaluate_model(pipe_base, "Base Model"))
print(evaluate_model(pipe_finetuned, "Fine-Tuned Model"))
