import torch
import numpy as np
from model import VAE
from utils import load_and_split_data, detect_anomaly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

_, X_test, _, _, scaler = load_and_split_data('../data/NF-UNSW-NB15.csv')

input_dim = X_test.shape[1]

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

model = VAE(input_dim).to(device)
model.load_state_dict(torch.load('../model.pth', weights_only=True))
model.eval()

errors = [detect_anomaly(model, x.unsqueeze(0), threshold=False)[1] for x in X_test_tensor[:100]]
threshold = np.percentile(errors, 95)

print("Monitoring real-time data...")
for i in range(len(X_test_tensor)):
    sample = X_test_tensor[i].unsqueeze(0)
    is_anomaly, anomaly_score = detect_anomaly(model, sample, threshold)

    if is_anomaly:
        print(f"Anomaly detected! Score: {anomaly_score:.4f}, Row: {i}")

