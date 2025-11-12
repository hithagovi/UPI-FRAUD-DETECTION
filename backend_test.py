#!/usr/bin/env python3

import requests
import sys
import json
import time
from datetime import datetime
import io
import pandas as pd

class FraudDetectionAPITester:
    def __init__(self, base_url="https://securitywatchdog.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.user_data = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
        
        result = {
            "test_name": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if details:
            print(f"    Details: {details}")

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, params=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        if files:
            # Remove Content-Type for file uploads
            headers.pop('Content-Type', None)

        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                if files:
                    response = requests.post(url, headers=headers, files=files, data=data)
                else:
                    response = requests.post(url, json=data, headers=headers)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, params=params)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)

            success = response.status_code == expected_status
            details = f"Status: {response.status_code}"
            
            if success and response.content:
                try:
                    response_data = response.json()
                    details += f", Response: {json.dumps(response_data, indent=2)[:200]}..."
                    self.log_test(name, True, details)
                    return True, response_data
                except:
                    self.log_test(name, True, details)
                    return True, {}
            elif not success:
                try:
                    error_data = response.json()
                    details += f", Error: {error_data}"
                except:
                    details += f", Error: {response.text[:200]}"
                self.log_test(name, False, details)
                return False, {}
            else:
                self.log_test(name, True, details)
                return True, {}

        except Exception as e:
            self.log_test(name, False, f"Exception: {str(e)}")
            return False, {}

    def test_user_registration(self):
        """Test user registration"""
        print("\n🔍 Testing User Registration...")
        
        # Test analyst registration
        analyst_data = {
            "email": f"analyst_{int(time.time())}@test.com",
            "password": "TestPass123!",
            "name": "Test Analyst",
            "role": "analyst"
        }
        
        success, response = self.run_test(
            "Register Analyst User",
            "POST",
            "auth/register",
            200,
            data=analyst_data
        )
        
        if success and 'token' in response:
            self.token = response['token']
            self.user_data = response['user']
            
            # Test admin registration
            admin_data = {
                "email": f"admin_{int(time.time())}@test.com",
                "password": "TestPass123!",
                "name": "Test Admin",
                "role": "admin"
            }
            
            self.run_test(
                "Register Admin User",
                "POST",
                "auth/register",
                200,
                data=admin_data
            )
            
            return True
        return False

    def test_user_login(self):
        """Test user login"""
        print("\n🔍 Testing User Login...")
        
        if not self.user_data:
            return False
            
        login_data = {
            "email": self.user_data['email'],
            "password": "TestPass123!"
        }
        
        success, response = self.run_test(
            "User Login",
            "POST",
            "auth/login",
            200,
            data=login_data
        )
        
        if success and 'token' in response:
            self.token = response['token']
            return True
        return False

    def test_auth_me(self):
        """Test get current user"""
        print("\n🔍 Testing Auth Me...")
        
        success, response = self.run_test(
            "Get Current User",
            "GET",
            "auth/me",
            200
        )
        return success

    def test_dashboard_metrics(self):
        """Test dashboard metrics"""
        print("\n🔍 Testing Dashboard Metrics...")
        
        success, response = self.run_test(
            "Get Dashboard Metrics",
            "GET",
            "dashboard/metrics",
            200
        )
        return success

    def test_dataset_upload(self):
        """Test dataset upload"""
        print("\n🔍 Testing Dataset Upload...")
        
        # Create a sample CSV dataset
        sample_data = {
            'amount': [100, 200, 50, 1000, 25],
            'sender': ['user1@upi', 'user2@upi', 'user3@upi', 'user4@upi', 'user5@upi'],
            'receiver': ['merchant1', 'merchant2', 'merchant3', 'merchant4', 'merchant5'],
            'is_fraud': [0, 0, 0, 1, 0]
        }
        
        df = pd.DataFrame(sample_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        files = {
            'file': ('test_dataset.csv', csv_content, 'text/csv')
        }
        
        data = {
            'fraud_column': 'is_fraud'
        }
        
        success, response = self.run_test(
            "Upload Dataset",
            "POST",
            "datasets/upload",
            200,
            data=data,
            files=files
        )
        
        if success and 'id' in response:
            self.dataset_id = response['id']
            return True
        return False

    def test_model_training(self):
        """Test model training"""
        print("\n🔍 Testing Model Training...")
        
        if not hasattr(self, 'dataset_id'):
            print("❌ No dataset available for training")
            return False
        
        success, response = self.run_test(
            "Train XGBoost Model",
            "POST",
            f"datasets/{self.dataset_id}/train?model_type=xgboost",
            200
        )
        
        if success:
            # Wait a moment for model to be loaded
            time.sleep(2)
            return True
        return False

    def test_get_active_model(self):
        """Test get active model"""
        print("\n🔍 Testing Get Active Model...")
        
        success, response = self.run_test(
            "Get Active Model",
            "GET",
            "models/active",
            200
        )
        return success

    def test_transaction_prediction(self):
        """Test transaction prediction"""
        print("\n🔍 Testing Transaction Prediction...")
        
        transaction_data = {
            'amount': 500,
            'sender': 'test@upi',
            'receiver': 'merchant_test'
        }
        
        success, response = self.run_test(
            "Predict Transaction",
            "POST",
            "transactions/predict",
            200,
            data=transaction_data
        )
        
        if success and 'id' in response:
            self.transaction_id = response['id']
            return True
        return False

    def test_transactions_api(self):
        """Test transactions API"""
        print("\n🔍 Testing Transactions API...")
        
        # Test get all transactions
        success1, _ = self.run_test(
            "Get All Transactions",
            "GET",
            "transactions",
            200
        )
        
        # Test filtered transactions
        success2, _ = self.run_test(
            "Get Fraudulent Transactions",
            "GET",
            "transactions",
            200,
            params={'filter_by': 'fraudulent', 'limit': 50}
        )
        
        # Test get specific transaction
        success3 = True
        if hasattr(self, 'transaction_id'):
            success3, _ = self.run_test(
                "Get Specific Transaction",
                "GET",
                f"transactions/{self.transaction_id}",
                200
            )
        
        return success1 and success2 and success3

    def test_analytics_api(self):
        """Test analytics API"""
        print("\n🔍 Testing Analytics API...")
        
        success, response = self.run_test(
            "Get Analytics Metrics",
            "GET",
            "analytics/metrics",
            200
        )
        return success

    def test_alerts_api(self):
        """Test alerts API"""
        print("\n🔍 Testing Alerts API...")
        
        # Test get alerts
        success1, response = self.run_test(
            "Get Alerts",
            "GET",
            "alerts",
            200,
            params={'limit': 10}
        )
        
        # Test resolve alert if any exist
        success2 = True
        if success1 and response and len(response) > 0:
            alert_id = response[0]['id']
            success2, _ = self.run_test(
                "Resolve Alert",
                "PUT",
                f"alerts/{alert_id}/resolve",
                200
            )
        
        return success1 and success2

    def test_block_management(self):
        """Test block management"""
        print("\n🔍 Testing Block Management...")
        
        # Test get blocked entities
        success1, _ = self.run_test(
            "Get Blocked Entities",
            "GET",
            "blocks",
            200
        )
        
        # Test block entity
        block_data = {
            "entity_type": "upi_id",
            "entity_value": "test_fraudster@upi",
            "reason": "Automated test block"
        }
        
        success2, response = self.run_test(
            "Block Entity",
            "POST",
            "blocks",
            200,
            data=block_data
        )
        
        # Test unblock entity
        success3 = True
        if success2 and 'id' in response:
            block_id = response['id']
            success3, _ = self.run_test(
                "Unblock Entity",
                "PUT",
                f"blocks/{block_id}/unblock",
                200,
                params={'reason': 'Automated test unblock'}
            )
        
        return success1 and success2 and success3

    def test_audit_logs(self):
        """Test audit logs (admin only)"""
        print("\n🔍 Testing Audit Logs...")
        
        # This might fail if user is not admin, which is expected
        success, _ = self.run_test(
            "Get Audit Logs",
            "GET",
            "audit-logs",
            200,
            params={'limit': 10}
        )
        
        # Don't fail the test if user is not admin
        return True

    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Fraud Detection API Tests...")
        print(f"📍 Testing against: {self.base_url}")
        
        # Authentication tests
        if not self.test_user_registration():
            print("❌ Registration failed, stopping tests")
            return False
            
        if not self.test_user_login():
            print("❌ Login failed, stopping tests")
            return False
            
        self.test_auth_me()
        
        # Core functionality tests
        self.test_dashboard_metrics()
        
        # Dataset and ML tests
        if self.test_dataset_upload():
            self.test_model_training()
            self.test_get_active_model()
            self.test_transaction_prediction()
        
        # API tests
        self.test_transactions_api()
        self.test_analytics_api()
        self.test_alerts_api()
        self.test_block_management()
        self.test_audit_logs()
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Tests Passed: {self.tests_passed}")
        print(f"   Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        return self.tests_passed == self.tests_run

def main():
    tester = FraudDetectionAPITester()
    success = tester.run_all_tests()
    
    # Save detailed results
    with open('/app/backend_test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'tests_run': tester.tests_run,
                'tests_passed': tester.tests_passed,
                'success_rate': (tester.tests_passed/tester.tests_run*100) if tester.tests_run > 0 else 0
            },
            'results': tester.test_results
        }, f, indent=2)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())