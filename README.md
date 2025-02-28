# SentinelADS: Real-Time Network Anomaly Detection System

SentinelADS is an AI-powered anomaly detection system designed to monitor network traffic and identify suspicious behavior in real time. It leverages a Variational Autoencoder (VAE) to learn normal network activity and detect deviations that may indicate cyber threats.

---

## How SentinelADS Works

### 1. Learning Normal Traffic Patterns
- SentinelADS is trained on network traffic data using a VAE, an unsupervised deep learning model.  
- The VAE learns to compress network features into a latent space and reconstruct them with minimal error.  
- During training, it only sees normal traffic, so anything that deviates significantly during inference is flagged as an anomaly.  

### 2. Real-Time Monitoring & Detection
- After training, SentinelADS continuously monitors live network data.  
- Each new traffic record is fed into the VAE, which attempts to reconstruct it.  
- If the reconstruction error is significantly higher than expected, the traffic is flagged as anomalous.  

### 3. Threshold-Based Alerting
- The anomaly threshold is set using the 95th percentile of normal reconstruction errors.  
- If an incoming traffic pattern has an error above this threshold, it is labeled suspicious.  

---

## Dataset: UNSW-NB15

SentinelADS is trained on the **[UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)**, 
a real-world dataset for network intrusion detection. It includes both normal and malicious traffic samples, 
with various network flow features.
