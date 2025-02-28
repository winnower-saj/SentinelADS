import subprocess

if __name__ == "__main__":
    print("Training the model...")
    subprocess.run(["python", "src/train.py"])
    
    print("Starting real-time anomaly detection...")
    subprocess.run(["python", "src/detect.py"])
