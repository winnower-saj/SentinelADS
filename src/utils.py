import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath, test_size=0.2):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack'], errors='ignore')
    X = df.drop(columns=['Label'], errors='ignore')
    y = df['Label'] if 'Label' in df.columns else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, scaler


def detect_anomaly(model, sample, threshold):
    with torch.no_grad():
        sample = sample.unsqueeze(0) if sample.dim() == 1 else sample
        reconstructed, _, _ = model(sample)
        error = torch.mean(torch.abs(sample - reconstructed)).item()
        return error > threshold, error
