from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import jwt
import bcrypt
import io
import json
from collections import Counter

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-this')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

security = HTTPBearer()

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Models Directory
MODELS_DIR = ROOT_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

DATASETS_DIR = ROOT_DIR / 'datasets'
DATASETS_DIR.mkdir(exist_ok=True)

# Pydantic Models
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    name: str
    role: str = "analyst"  # admin or analyst
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = "analyst"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Transaction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transaction_data: Dict[str, Any]
    prediction: str  # "Fraudulent", "Suspicious", "Safe"
    fraud_probability: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    flagged: bool = False

class BlockedEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: str  # "upi_id" or "merchant"
    entity_value: str
    reason: str
    blocked_by: str
    blocked_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True

class BlockEntityRequest(BaseModel):
    entity_type: str
    entity_value: str
    reason: str

class Alert(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str
    alert_type: str  # "high_risk", "blocked_entity", "unusual_pattern"
    message: str
    severity: str  # "high", "medium", "low"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False

class AuditLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action: str
    performed_by: str
    entity_type: str
    entity_value: str
    details: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    uploaded_by: str
    uploaded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rows: int
    columns: List[str]
    fraud_column: Optional[str] = None
    status: str = "uploaded"  # uploaded, training, trained, failed

class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str
    model_type: str  # "xgboost" or "lightgbm"
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    trained_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True

# Global model variables
active_model = None
active_scaler = None
active_feature_columns = None
label_encoders = {}

# Authentication helpers
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_email = payload.get('email')
        if not user_email:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_doc = await db.users.find_one({'email': user_email}, {'_id': 0, 'password': 0})
        if not user_doc:
            raise HTTPException(status_code=401, detail="User not found")
        
        return User(**user_doc)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def admin_only(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# Auth endpoints
@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    existing = await db.users.find_one({'email': user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
    
    user = User(
        email=user_data.email,
        name=user_data.name,
        role=user_data.role
    )
    
    user_dict = user.model_dump()
    user_dict['password'] = hashed_password.decode('utf-8')
    
    await db.users.insert_one(user_dict)
    
    token = jwt.encode(
        {'email': user.email, 'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )
    
    return {'token': token, 'user': user}

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user_doc = await db.users.find_one({'email': credentials.email})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not bcrypt.checkpw(credentials.password.encode('utf-8'), user_doc['password'].encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = jwt.encode(
        {'email': credentials.email, 'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )
    
    user_doc.pop('password')
    user_doc.pop('_id')
    
    return {'token': token, 'user': User(**user_doc)}

@api_router.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# Dataset upload and training
@api_router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), fraud_column: str = "is_fraud", current_user: User = Depends(get_current_user)):
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported")
        
        if fraud_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Fraud column '{fraud_column}' not found in dataset")
        
        dataset_id = str(uuid.uuid4())
        filepath = DATASETS_DIR / f"{dataset_id}.parquet"
        df.to_parquet(filepath)
        
        metadata = DatasetMetadata(
            id=dataset_id,
            filename=file.filename,
            uploaded_by=current_user.email,
            rows=len(df),
            columns=df.columns.tolist(),
            fraud_column=fraud_column,
            status="uploaded"
        )
        
        await db.datasets.insert_one(metadata.model_dump())
        
        return metadata
    except Exception as e:
        logging.error(f"Dataset upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")

@api_router.post("/datasets/{dataset_id}/train")
async def train_model(dataset_id: str, model_type: str = "xgboost", current_user: User = Depends(get_current_user)):
    try:
        dataset_doc = await db.datasets.find_one({'id': dataset_id}, {'_id': 0})
        if not dataset_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset = DatasetMetadata(**dataset_doc)
        await db.datasets.update_one({'id': dataset_id}, {'$set': {'status': 'training'}})
        
        filepath = DATASETS_DIR / f"{dataset_id}.parquet"
        df = pd.read_parquet(filepath)
        
        fraud_col = dataset.fraud_column
        if fraud_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Fraud column '{fraud_col}' not found")
        
        # Prepare features and target
        X = df.drop(columns=[fraud_col])
        y = df[fraud_col]
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        global label_encoders
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Handle imbalanced data with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == "xgboost":
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        model.fit(X_train_scaled, y_train_balanced)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': float(model.score(X_test_scaled, y_test)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        # Feature importance
        feature_importance = dict(zip(X.columns.tolist(), model.feature_importances_.tolist()))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Save model
        model_id = str(uuid.uuid4())
        model_path = MODELS_DIR / f"{model_id}.joblib"
        scaler_path = MODELS_DIR / f"{model_id}_scaler.joblib"
        encoders_path = MODELS_DIR / f"{model_id}_encoders.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoders, encoders_path)
        
        # Save metadata
        model_metadata = ModelMetadata(
            id=model_id,
            dataset_id=dataset_id,
            model_type=model_type,
            metrics=metrics,
            feature_importance=feature_importance
        )
        
        # Deactivate old models
        await db.models.update_many({}, {'$set': {'is_active': False}})
        
        # Save new model
        await db.models.insert_one(model_metadata.model_dump())
        await db.datasets.update_one({'id': dataset_id}, {'$set': {'status': 'trained'}})
        
        # Load model into memory
        global active_model, active_scaler, active_feature_columns
        active_model = model
        active_scaler = scaler
        active_feature_columns = X.columns.tolist()
        
        return model_metadata
    except Exception as e:
        await db.datasets.update_one({'id': dataset_id}, {'$set': {'status': 'failed'}})
        logging.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@api_router.get("/models/active")
async def get_active_model(current_user: User = Depends(get_current_user)):
    model_doc = await db.models.find_one({'is_active': True}, {'_id': 0})
    if not model_doc:
        raise HTTPException(status_code=404, detail="No active model found. Please upload and train a dataset first.")
    return ModelMetadata(**model_doc)

# Load active model on startup
@app.on_event("startup")
async def load_active_model():
    global active_model, active_scaler, active_feature_columns, label_encoders
    try:
        model_doc = await db.models.find_one({'is_active': True}, {'_id': 0})
        if model_doc:
            model_id = model_doc['id']
            model_path = MODELS_DIR / f"{model_id}.joblib"
            scaler_path = MODELS_DIR / f"{model_id}_scaler.joblib"
            encoders_path = MODELS_DIR / f"{model_id}_encoders.joblib"
            
            if model_path.exists():
                active_model = joblib.load(model_path)
                active_scaler = joblib.load(scaler_path)
                label_encoders = joblib.load(encoders_path)
                
                dataset_doc = await db.datasets.find_one({'id': model_doc['dataset_id']}, {'_id': 0})
                if dataset_doc:
                    dataset = DatasetMetadata(**dataset_doc)
                    active_feature_columns = [col for col in dataset.columns if col != dataset.fraud_column]
                
                logging.info("Active model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading active model: {str(e)}")

# Prediction endpoint
@api_router.post("/transactions/predict")
async def predict_transaction(transaction_data: Dict[str, Any], current_user: User = Depends(get_current_user)):
    global active_model, active_scaler, active_feature_columns, label_encoders
    
    if active_model is None:
        raise HTTPException(status_code=400, detail="No trained model available. Please train a model first.")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Apply label encoders
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except:
                    df[col] = 0  # Unknown category
        
        # Ensure all required columns exist
        for col in active_feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[active_feature_columns]
        df = df.fillna(df.mean())
        
        # Scale and predict
        X_scaled = active_scaler.transform(df)
        prediction_proba = active_model.predict_proba(X_scaled)[0]
        fraud_probability = float(prediction_proba[1])
        
        # Classify
        if fraud_probability >= 0.7:
            prediction = "Fraudulent"
            severity = "high"
        elif fraud_probability >= 0.4:
            prediction = "Suspicious"
            severity = "medium"
        else:
            prediction = "Safe"
            severity = "low"
        
        # Save transaction
        transaction = Transaction(
            transaction_data=transaction_data,
            prediction=prediction,
            fraud_probability=fraud_probability,
            flagged=(prediction in ["Fraudulent", "Suspicious"])
        )
        
        await db.transactions.insert_one(transaction.model_dump())
        
        # Create alert if fraud detected
        if prediction in ["Fraudulent", "Suspicious"]:
            alert = Alert(
                transaction_id=transaction.id,
                alert_type="high_risk" if prediction == "Fraudulent" else "unusual_pattern",
                message=f"{prediction} transaction detected with {fraud_probability*100:.1f}% confidence",
                severity=severity
            )
            await db.alerts.insert_one(alert.model_dump())
        
        return transaction
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Transactions endpoints
@api_router.get("/transactions")
async def get_transactions(
    filter_by: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    query = {}
    if filter_by and filter_by != "all":
        query['prediction'] = filter_by.capitalize()
    
    transactions = await db.transactions.find(query, {'_id': 0}).sort('timestamp', -1).limit(limit).to_list(limit)
    return transactions

@api_router.get("/transactions/{transaction_id}")
async def get_transaction(transaction_id: str, current_user: User = Depends(get_current_user)):
    transaction = await db.transactions.find_one({'id': transaction_id}, {'_id': 0})
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction

# Dashboard metrics
@api_router.get("/dashboard/metrics")
async def get_dashboard_metrics(current_user: User = Depends(get_current_user)):
    try:
        total_transactions = await db.transactions.count_documents({})
        fraudulent_count = await db.transactions.count_documents({'prediction': 'Fraudulent'})
        suspicious_count = await db.transactions.count_documents({'prediction': 'Suspicious'})
        active_alerts = await db.alerts.count_documents({'resolved': False})
        blocked_entities = await db.blocked_entities.count_documents({'is_active': True})
        
        fraud_rate = (fraudulent_count / total_transactions * 100) if total_transactions > 0 else 0
        
        return {
            'total_transactions': total_transactions,
            'fraudulent_count': fraudulent_count,
            'suspicious_count': suspicious_count,
            'active_alerts': active_alerts,
            'blocked_entities': blocked_entities,
            'fraud_detection_rate': round(fraud_rate, 2)
        }
    except Exception as e:
        logging.error(f"Dashboard metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@api_router.get("/analytics/metrics")
async def get_analytics(current_user: User = Depends(get_current_user)):
    try:
        transactions = await db.transactions.find({}, {'_id': 0}).to_list(10000)
        
        if not transactions:
            return {
                'avg_transaction_value': 0,
                'avg_fraud_amount': 0,
                'fraud_detection_rate': 0,
                'top_senders': [],
                'amount_distribution': []
            }
        
        df = pd.DataFrame(transactions)
        
        # Calculate metrics
        amounts = [t.get('transaction_data', {}).get('amount', 0) for t in transactions]
        avg_transaction_value = np.mean(amounts) if amounts else 0
        
        fraud_amounts = [t.get('transaction_data', {}).get('amount', 0) for t in transactions if t.get('prediction') == 'Fraudulent']
        avg_fraud_amount = np.mean(fraud_amounts) if fraud_amounts else 0
        
        total = len(transactions)
        fraudulent = len([t for t in transactions if t.get('prediction') == 'Fraudulent'])
        fraud_rate = (fraudulent / total * 100) if total > 0 else 0
        
        # Top senders by volume
        senders = [t.get('transaction_data', {}).get('sender', 'Unknown') for t in transactions]
        sender_counts = Counter(senders).most_common(5)
        top_senders = [{'sender': s, 'count': c} for s, c in sender_counts]
        
        # Amount distribution
        amount_ranges = [
            {'range': '0-100', 'count': len([a for a in amounts if 0 <= a <= 100])},
            {'range': '101-500', 'count': len([a for a in amounts if 101 <= a <= 500])},
            {'range': '501-1000', 'count': len([a for a in amounts if 501 <= a <= 1000])},
            {'range': '1001-5000', 'count': len([a for a in amounts if 1001 <= a <= 5000])},
            {'range': '5000+', 'count': len([a for a in amounts if a > 5000])}
        ]
        
        return {
            'avg_transaction_value': round(avg_transaction_value, 2),
            'avg_fraud_amount': round(avg_fraud_amount, 2),
            'fraud_detection_rate': round(fraud_rate, 2),
            'top_senders': top_senders,
            'amount_distribution': amount_ranges
        }
    except Exception as e:
        logging.error(f"Analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Alerts endpoints
@api_router.get("/alerts")
async def get_alerts(limit: int = 50, current_user: User = Depends(get_current_user)):
    alerts = await db.alerts.find({}, {'_id': 0}).sort('created_at', -1).limit(limit).to_list(limit)
    return alerts

@api_router.put("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, current_user: User = Depends(get_current_user)):
    result = await db.alerts.update_one({'id': alert_id}, {'$set': {'resolved': True}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {'message': 'Alert resolved'}

# Block Management endpoints
@api_router.get("/blocks")
async def get_blocked_entities(current_user: User = Depends(get_current_user)):
    blocks = await db.blocked_entities.find({'is_active': True}, {'_id': 0}).sort('blocked_at', -1).to_list(1000)
    return blocks

@api_router.post("/blocks")
async def block_entity(request: BlockEntityRequest, current_user: User = Depends(get_current_user)):
    existing = await db.blocked_entities.find_one({
        'entity_type': request.entity_type,
        'entity_value': request.entity_value,
        'is_active': True
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="Entity already blocked")
    
    blocked = BlockedEntity(
        entity_type=request.entity_type,
        entity_value=request.entity_value,
        reason=request.reason,
        blocked_by=current_user.email
    )
    
    await db.blocked_entities.insert_one(blocked.model_dump())
    
    # Create audit log
    audit = AuditLog(
        action="block",
        performed_by=current_user.email,
        entity_type=request.entity_type,
        entity_value=request.entity_value,
        details=f"Blocked: {request.reason}"
    )
    await db.audit_logs.insert_one(audit.model_dump())
    
    return blocked

@api_router.put("/blocks/{block_id}/unblock")
async def unblock_entity(block_id: str, reason: str, current_user: User = Depends(get_current_user)):
    block_doc = await db.blocked_entities.find_one({'id': block_id})
    if not block_doc:
        raise HTTPException(status_code=404, detail="Blocked entity not found")
    
    result = await db.blocked_entities.update_one({'id': block_id}, {'$set': {'is_active': False}})
    
    # Create audit log
    audit = AuditLog(
        action="unblock",
        performed_by=current_user.email,
        entity_type=block_doc['entity_type'],
        entity_value=block_doc['entity_value'],
        details=f"Unblocked: {reason}"
    )
    await db.audit_logs.insert_one(audit.model_dump())
    
    return {'message': 'Entity unblocked'}

# Audit logs
@api_router.get("/audit-logs")
async def get_audit_logs(limit: int = 100, current_user: User = Depends(admin_only)):
    logs = await db.audit_logs.find({}, {'_id': 0}).sort('timestamp', -1).limit(limit).to_list(limit)
    return logs

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
