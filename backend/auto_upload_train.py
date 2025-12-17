import requests
import os

BASE_URL = "http://127.0.0.1:8000"
EMAIL = "hithagovi@gmail.com"        # your registered email
PASSWORD = "hitha"      # exact password
DATASETS_FOLDER = "./datasets"       # folder with your CSV files
MODEL_TYPE = "xgboost"
FRAUD_COLUMN_NAME = "is_fraud"

# ===== STEP 1: LOGIN =====
login_url = f"{BASE_URL}/api/auth/login"
login_payload = {"email": EMAIL, "password": PASSWORD}

resp = requests.post(login_url, json=login_payload)
if resp.status_code != 200:
    print("❌ Login failed:", resp.text)
    exit(1)

token = resp.json()["token"]
headers = {"Authorization": f"Bearer {token}"}
print("✅ Logged in successfully.")

# ===== STEP 2: UPLOAD & TRAIN =====
for file_name in os.listdir(DATASETS_FOLDER):
    if not file_name.endswith(".csv"):
        continue

    csv_path = os.path.join(DATASETS_FOLDER, file_name)
    print(f"\n📤 Uploading dataset: {file_name}")

    upload_url = f"{BASE_URL}/api/datasets/upload?fraud_column={FRAUD_COLUMN_NAME}"
    with open(csv_path, "rb") as f:
        files = {"file": (file_name, f, "text/csv")}
        upload_resp = requests.post(upload_url, headers=headers, files=files)
    
    if upload_resp.status_code != 200:
        print("❌ Upload failed:", upload_resp.text)
        continue

    dataset_id = upload_resp.json()["id"]
    print(f"✅ Dataset uploaded. ID: {dataset_id}")

    # Train model
    train_url = f"{BASE_URL}/api/datasets/{dataset_id}/train?model_type={MODEL_TYPE}"
    train_resp = requests.post(train_url, headers=headers)
    if train_resp.status_code != 200:
        print("❌ Training failed:", train_resp.text)
        continue

    metrics = train_resp.json()
    print(f"✅ Model trained for {file_name}. Metrics:")
    print(metrics)
