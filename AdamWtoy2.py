import torch
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
import copy

# ---- Step 1. Create nonlinear dataset ----
torch.manual_seed(0)
x = torch.linspace(-1, 1, 400).unsqueeze(1)
y = torch.sin(2 * torch.pi * x) + 0.1 * torch.randn_like(x)

# ---- Step 2. Model creation ----
def create_model():
    # Slightly wider MLP to handle nonlinear mapping
    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )
    return model

# ---- Step 3. Training + evaluation ----
def train_model(model, optimizer, x, y, epochs=2000):
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate(model, x, y):
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        y_pred = model(x)
        return loss_fn(y_pred, y).item()

# ---- Step 4. Random restarts ----
best_val_loss = float("inf")
best_model_state = None

for seed in range(50):
    torch.manual_seed(seed)
    model = create_model()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    train_loss = train_model(model, optimizer, x, y)
    val_loss = evaluate(model, x, y)
    print(f"Run {seed+1}: Final train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())

# ---- Step 5. Visualize the best model ----
best_model = create_model()
best_model.load_state_dict(best_model_state)

with torch.no_grad():
    y_pred = best_model(x)

plt.figure(figsize=(8,4))
plt.scatter(x, y, s=10, label="Data", alpha=0.6)
plt.plot(x, y_pred, color='red', label="Best AdamW model")
plt.legend()
plt.title(f"Best model (val_loss={best_val_loss:.4f})")
plt.show()
