# server.py (fixed for working login/register and training)
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os, logging, uuid, io, jwt, bcrypt, joblib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from pydantic import BaseModel, Field, EmailStr, ConfigDict
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Paths & Env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')
MONGO_URL = os.getenv('MONGO_URL')
DB_NAME = os.getenv('DB_NAME')
JWT_SECRET = os.getenv('JWT_SECRET', 'secret')
JWT_ALGO = 'HS256'
JWT_EXP_HOURS = 24

MODELS_DIR = ROOT_DIR / 'models'
DATASETS_DIR = ROOT_DIR / 'datasets'
MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

# App
app = FastAPI()
router = APIRouter(prefix="/api")
security = HTTPBearer()

client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=2000)
db = client[DB_NAME] if client else None

# In-memory fallback
IN_MEM_USERS = {}
IN_MEM_DATASETS = {}
IN_MEM_MODELS = {}

# Models
class User(BaseModel):
    model_config = ConfigDict(extra='ignore')
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    name: str
    role: str = 'analyst'
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = 'analyst'

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra='ignore')
    id: str
    filename: str
    uploaded_by: str
    uploaded_at: str
    rows: int
    columns: List[str]
    fraud_column: str
    status: str

class ModelMetadata(BaseModel):
    id: str
    dataset_id: str
    model_type: str
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    trained_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True

# Auth helper
async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGO])
        email = payload.get('email')
        if db:
            user = await db.users.find_one({'email': email})
        else:
            user = next((u for u in IN_MEM_USERS.values() if u['email'] == email), None)
        if not user:
            raise HTTPException(401, 'User not found')
        return User(**user)
    except Exception:
        raise HTTPException(401, 'Invalid token')

# Register
@router.post('/auth/register')
async def register(data: UserRegister):
    if db:
        exists = await db.users.find_one({'email': data.email})
        if exists: raise HTTPException(400, 'Email exists')
    else:
        if any(u['email']==data.email for u in IN_MEM_USERS.values()):
            raise HTTPException(400, 'Email exists')

    hashed = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    user = User(email=data.email, name=data.name, role=data.role).model_dump()
    user['password'] = hashed

    if db:
        await db.users.insert_one(user)
    else:
        IN_MEM_USERS[user['id']] = user

    token = jwt.encode({'email': data.email, 'exp': datetime.utcnow() + timedelta(hours=JWT_EXP_HOURS)}, JWT_SECRET, algorithm=JWT_ALGO)
    return {'token': token, 'user': {k:v for k,v in user.items() if k != 'password'}}

# Login
@router.post('/auth/login')
async def login(data: UserLogin):
    if db:
        user = await db.users.find_one({'email': data.email})
    else:
        user = next((u for u in IN_MEM_USERS.values() if u['email']==data.email), None)
    if not user or not bcrypt.checkpw(data.password.encode(), user['password'].encode()):
        raise HTTPException(401, 'Invalid credentials')
    token = jwt.encode({'email': data.email, 'exp': datetime.utcnow() + timedelta(hours=JWT_EXP_HOURS)}, JWT_SECRET, algorithm=JWT_ALGO)
    return {'token': token, 'user': {k:v for k,v in user.items() if k != 'password'}}

# Upload dataset
@router.post('/datasets/upload')
async def upload_dataset(file: UploadFile = File(...), fraud_column: str = 'is_fraud', user: User = Depends(get_current_user)):
    data = await file.read()
    df = pd.read_csv(io.BytesIO(data))
    if fraud_column not in df.columns:
        raise HTTPException(400, 'Fraud column missing')
    dataset_id = str(uuid.uuid4())
    df.to_parquet(DATASETS_DIR / f'{dataset_id}.parquet')
    meta = DatasetMetadata(
        id=dataset_id, filename=file.filename, uploaded_by=user.email,
        uploaded_at=datetime.utcnow().isoformat(), rows=len(df),
        columns=df.columns.tolist(), fraud_column=fraud_column, status='uploaded'
    ).model_dump()
    if db:
        await db.datasets.insert_one(meta)
    else:
        IN_MEM_DATASETS[dataset_id] = meta
    return meta

# Train dataset
@router.post('/datasets/{dataset_id}/train')
async def train(dataset_id: str, user: User = Depends(get_current_user)):
    if db:
        ds = await db.datasets.find_one({'id': dataset_id})
        if not ds: raise HTTPException(404, 'Dataset not found')
        await db.datasets.update_one({'id': dataset_id}, {'$set': {'status': 'training'}})
    else:
        ds = IN_MEM_DATASETS.get(dataset_id)
        if not ds: raise HTTPException(404, 'Dataset not found')
        ds['status'] = 'training'

    df = pd.read_parquet(DATASETS_DIR / f'{dataset_id}.parquet')
    X = df.drop(columns=[ds['fraud_column']])
    y = df[ds['fraud_column']]

    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X = X.fillna(X.mean())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:,1]

    metrics = {
        'precision': float(precision_score(yte, preds)),
        'recall': float(recall_score(yte, preds)),
        'f1': float(f1_score(yte, preds)),
        'roc_auc': float(roc_auc_score(yte, probs))
    }

    mid = str(uuid.uuid4())
    joblib.dump(model, MODELS_DIR / f'{mid}.joblib')
    joblib.dump(scaler, MODELS_DIR / f'{mid}_scaler.joblib')
    joblib.dump(encoders, MODELS_DIR / f'{mid}_encoders.joblib')

    if db:
        await db.models.update_many({}, {'$set': {'is_active': False}})
        await db.models.insert_one(ModelMetadata(id=mid, dataset_id=dataset_id, model_type='xgboost',
                                                 metrics=metrics,
                                                 feature_importance=dict(zip(X.columns, model.feature_importances_))).model_dump())
        await db.datasets.update_one({'id': dataset_id}, {'$set': {'status': 'trained'}})
    else:
        IN_MEM_MODELS[mid] = metrics
        ds['status'] = 'trained'

    return {'model_id': mid, 'metrics': metrics}

# Router & CORS
app.include_router(router)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
logging.basicConfig(level=logging.INFO)
