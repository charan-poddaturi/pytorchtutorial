import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Data Loading ──────────────────────────────────────────────────────────────
df = pd.read_csv('breast-cancer.csv')
df.drop(columns=['id'], inplace=True)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # fix random_state for reproducibility
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

x_train_tensor = torch.from_numpy(x_train).float()
x_test_tensor  = torch.from_numpy(x_test).float()
y_train_tensor = torch.from_numpy(y_train).float()
y_test_tensor  = torch.from_numpy(y_test).float()


# ── Improved Model ────────────────────────────────────────────────────────────
class MySimpleNN:
    def __init__(self, x):
        n_features = x.shape[1]

        # FIX 1: Xavier initialization instead of torch.rand (0–1)
        std = np.sqrt(2.0 / n_features)
        self.weights = torch.randn(n_features, 1, dtype=torch.float32) * std
        self.weights.requires_grad_(True)
        self.bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return torch.sigmoid(torch.matmul(x, self.weights) + self.bias)

    def loss(self, y_pred, y):
        # FIX 2: Use the `y` parameter, not the global y_train_tensor
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        y = y.view(-1, 1)  # reshape to match y_pred shape
        return -torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))


# ── Training ──────────────────────────────────────────────────────────────────
# FIX 3: More epochs + lower learning rate for stable convergence
learning_rate = 0.01
epochs        = 500

model = MySimpleNN(x_train_tensor)

loss_history = []
for epoch in range(epochs):
    y_pred = model.forward(x_train_tensor)
    current_loss = model.loss(y_pred, y_train_tensor)

    current_loss.backward()

    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias    -= learning_rate * model.bias.grad

    model.weights.grad.zero_()
    model.bias.grad.zero_()

    loss_history.append(current_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss.item():.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────
with torch.no_grad():
    # Test accuracy
    y_test_pred  = model.forward(x_test_tensor)
    y_test_labels = (y_test_pred > 0.5).float()
    test_acc = (y_test_labels.squeeze() == y_test_tensor).float().mean()
    print(f"\nTest Accuracy: {test_acc.item() * 100:.2f}%")

    # Train accuracy (to check for overfitting)
    y_train_pred   = model.forward(x_train_tensor)
    y_train_labels = (y_train_pred > 0.5).float()
    train_acc = (y_train_labels.squeeze() == y_train_tensor).float().mean()
    print(f"Train Accuracy: {train_acc.item() * 100:.2f}%")