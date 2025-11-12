#!/usr/bin/env python3

import requests
import pandas as pd
import numpy as np
import io

def create_better_dataset():
    """Create a larger, more balanced dataset for testing"""
    np.random.seed(42)
    
    # Create 100 transactions with balanced fraud/non-fraud
    n_samples = 100
    
    data = {
        'amount': np.random.uniform(10, 5000, n_samples),
        'sender_age': np.random.randint(18, 80, n_samples),
        'transaction_hour': np.random.randint(0, 24, n_samples),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'atm'], n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'previous_transactions': np.random.randint(0, 100, n_samples),
    }
    
    # Create fraud labels - 30% fraud rate
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    data['is_fraud'] = np.zeros(n_samples)
    data['is_fraud'][fraud_indices] = 1
    
    # Make fraudulent transactions more suspicious
    for idx in fraud_indices:
        if np.random.random() > 0.5:
            data['amount'][idx] = np.random.uniform(2000, 5000)  # Higher amounts
        if np.random.random() > 0.5:
            data['transaction_hour'][idx] = np.random.choice([0, 1, 2, 3, 22, 23])  # Odd hours
    
    df = pd.DataFrame(data)
    return df

def test_model_training_with_better_data():
    """Test model training with better dataset"""
    base_url = "https://securitywatchdog.preview.emergentagent.com/api"
    
    # First register and login
    register_data = {
        "email": f"tester_better_{int(pd.Timestamp.now().timestamp())}@test.com",
        "password": "TestPass123!",
        "name": "Better Test User",
        "role": "analyst"
    }
    
    response = requests.post(f"{base_url}/auth/register", json=register_data)
    if response.status_code != 200:
        print(f"❌ Registration failed: {response.text}")
        return False
    
    token = response.json()['token']
    headers = {'Authorization': f'Bearer {token}'}
    
    # Create better dataset
    df = create_better_dataset()
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    print(f"📊 Created dataset with {len(df)} rows, {df['is_fraud'].sum()} fraudulent transactions")
    
    # Upload dataset
    files = {
        'file': ('better_test_dataset.csv', csv_content, 'text/csv')
    }
    data = {'fraud_column': 'is_fraud'}
    
    response = requests.post(f"{base_url}/datasets/upload", files=files, data=data, headers={'Authorization': f'Bearer {token}'})
    if response.status_code != 200:
        print(f"❌ Dataset upload failed: {response.text}")
        return False
    
    dataset_id = response.json()['id']
    print(f"✅ Dataset uploaded successfully: {dataset_id}")
    
    # Train model
    response = requests.post(f"{base_url}/datasets/{dataset_id}/train?model_type=xgboost", headers=headers)
    if response.status_code != 200:
        print(f"❌ Model training failed: {response.text}")
        return False
    
    print("✅ Model trained successfully!")
    
    # Test prediction
    transaction_data = {
        'amount': 3000,
        'sender_age': 25,
        'transaction_hour': 2,
        'merchant_category': 'online',
        'is_weekend': 1,
        'previous_transactions': 5
    }
    
    response = requests.post(f"{base_url}/transactions/predict", json=transaction_data, headers=headers)
    if response.status_code != 200:
        print(f"❌ Prediction failed: {response.text}")
        return False
    
    prediction = response.json()
    print(f"✅ Prediction successful: {prediction['prediction']} ({prediction['fraud_probability']:.2%})")
    
    return True

if __name__ == "__main__":
    success = test_model_training_with_better_data()
    print(f"\n🎯 Better dataset test: {'PASSED' if success else 'FAILED'}")