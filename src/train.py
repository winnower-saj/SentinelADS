import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import VAE
from utils import load_and_split_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train, X_test, y_train, y_test, scaler = load_and_split_data('../data/NF-UNSW-NB15.csv')
mask = y_train == 0
X_train = X_train[mask]
y_train = y_train[mask]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values if y_train is not None else [], dtype=torch.float32).to(device)

input_dim = X_train.shape[1]
model = VAE(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def vae_loss(reconstructed_x, x, mu, log_var):
    reconstruction_loss = nn.MSELoss()(reconstructed_x, x)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

epochs = 5
batch_size = 64

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)


for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        batch = batch[0].to(device)
        optimizer.zero_grad()
        outputs, mu, log_var = model(batch)
        loss = vae_loss(outputs, batch, mu, log_var)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), '../model.pth')
