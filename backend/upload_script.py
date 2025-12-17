import requests, json, sys
url = '"'http://127.0.0.1:8000/api/datasets/upload'"'
token = sys.argv[1]
files = {'file': ('sample_upload.csv', open('sample_upload.csv','rb'))}
data = {'fraud_column': 'is_fraud'}
headers = {'Authorization': f'Bearer {token}'}
r = requests.post(url, headers=headers, files=files, data=data)
print('UPLOAD_STATUS', r.status_code)
try:
    print(json.dumps(r.json(), indent=2))
except Exception:
    print(r.text)
