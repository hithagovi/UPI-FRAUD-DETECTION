import os
import pathlib
import json

# Ensure dev upload is enabled for this process
os.environ['DEV_UPLOAD_ENABLED'] = 'true'

from fastapi.testclient import TestClient
import runpy

# Load server.py as a module and grab the `app` object
module_globals = runpy.run_path(str(pathlib.Path(__file__).parent.parent / 'server.py'))
if 'app' not in module_globals:
    raise RuntimeError('Could not find `app` in server.py')
app = module_globals['app']

client = TestClient(app)

csv_path = pathlib.Path(__file__).parent.parent / 'sample_dev.csv'
if not csv_path.exists():
    # try fallback location
    csv_path = pathlib.Path('sample_dev.csv')

print('Using CSV:', csv_path)

with open(csv_path, 'rb') as fh:
    files = {'file': ('sample_dev.csv', fh, 'text/csv')}
    data = {'fraud_column': 'is_fraud'}
    resp = client.post('/api/dev/datasets/upload', files=files, data=data)

print('Status code:', resp.status_code)
try:
    print('Response JSON:', json.dumps(resp.json(), indent=2))
except Exception:
    print('Response text:', resp.text)

ds_dir = pathlib.Path(__file__).parent.parent / 'datasets'
if ds_dir.exists():
    files = list(ds_dir.glob('*.parquet'))
    print('Saved parquet files:', [f.name for f in files])
else:
    print('Datasets directory not found:', ds_dir)
