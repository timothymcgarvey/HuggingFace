import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt

# ---- Step 1. Generate dataset ----
x = np.linspace(-4*np.pi, 4*np.pi, 80)
y = np.linspace(-4*np.pi, 4*np.pi, 80)
X, Y = np.meshgrid(x, y)
Z = np.exp(-0.1*(X**2 + Y**2)) * np.sin(2*X) * np.cos(2*Y)

X_flat = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
Z_flat = torch.tensor(Z.ravel(), dtype=torch.float32).unsqueeze(1)

# ---- Step 2. Model definition ----
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---- Step 3. Training function ----
def train_model(model, optimizer, X_flat, Z_flat, epochs=8000):
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_flat)
        loss = loss_fn(y_pred, Z_flat)
        loss.backward()
        optimizer.step()
    return loss.item()

# ---- Step 4. Random restarts ----
num_restarts = 5
results = []

for seed in range(num_restarts):
    torch.manual_seed(seed)
    model = Net()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    final_loss = train_model(model, optimizer, X_flat, Z_flat)
    with torch.no_grad():
        pred = model(X_flat).reshape(Z.shape).numpy()
    results.append((seed, final_loss, pred))
    print(f"Restart {seed+1}/{num_restarts}: Final loss = {final_loss:.6f}")

# ---- Step 5. Plot true surface + all learned surfaces ----
fig = plt.figure(figsize=(15, 8))
ax_true = fig.add_subplot(2, 3, 1, projection='3d')
ax_true.plot_surface(X, Y, Z, cmap='viridis')
ax_true.set_title("True Surface")

for i, (seed, loss, pred) in enumerate(results, start=2):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    ax.plot_surface(X, Y, pred, cmap='viridis')
    ax.set_title(f"Restart {seed} (Loss={loss:.4f})")

plt.tight_layout()
plt.show()
