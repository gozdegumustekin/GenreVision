import os
import numpy as np
import torch
import torch.nn as nn
import requests
import boto3
from io import BytesIO
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import models, transforms
from PIL import Image

# === KONFIGÜRASYON (Railway environment variables) ===
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY', '').strip().strip('"').strip("'")
R2_SECRET_KEY = os.getenv('R2_SECRET_KEY', '').strip().strip('"').strip("'")
R2_ENDPOINT_URL = os.getenv('R2_ENDPOINT_URL', '').strip().strip('"').strip("'").rstrip('/')
R2_BUCKET = os.getenv('R2_BUCKET', 'genrevision').strip().strip('"').strip("'")

MODEL_PATH = 'genre_vision_v3_best.pth'
THRESHOLDS_PATH = 'best_thresholds_v3.npy'

app = FastAPI(title='GenreVision API v3')
app.add_middleware(CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

device = torch.device('cpu')
model = None
label_cols = None
thresholds = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PredictionRequest(BaseModel):
    url: str


def diagnose_config():
    issues = []
    if not R2_ACCESS_KEY:
        issues.append('R2_ACCESS_KEY boş')
    elif len(R2_ACCESS_KEY) != 32:
        issues.append(f'R2_ACCESS_KEY uzunluğu {len(R2_ACCESS_KEY)}, 32 olmalı')
    if not R2_SECRET_KEY:
        issues.append('R2_SECRET_KEY boş')
    elif len(R2_SECRET_KEY) != 64:
        issues.append(f'R2_SECRET_KEY uzunluğu {len(R2_SECRET_KEY)}, 64 olmalı')
    if not R2_ENDPOINT_URL:
        issues.append('R2_ENDPOINT_URL boş')
    elif not R2_ENDPOINT_URL.startswith('https://'):
        issues.append('R2_ENDPOINT_URL https:// ile başlamalı')
    if not R2_BUCKET:
        issues.append('R2_BUCKET boş')
    return issues


def download_from_r2(key: str, local_path: str):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        print(f'  {local_path} zaten var, atlandı')
        return
    s3 = boto3.client('s3', endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY,
        region_name='auto')
    try:
        s3.list_objects_v2(Bucket=R2_BUCKET, Prefix='model/', MaxKeys=10)
    except Exception as e:
        raise RuntimeError(f'R2 bucket erişimi yok: {e}')
    s3.download_file(R2_BUCKET, key, local_path)
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f'  ✓ {key} indirildi ({size_mb:.2f} MB)')


@app.on_event('startup')
def load_model():
    global model, label_cols, thresholds
    print('=' * 60)
    print('Startup: Konfigürasyon kontrol ediliyor...')

    issues = diagnose_config()
    if issues:
        for issue in issues:
            print(f'  ✗ {issue}')
        raise RuntimeError(f'Env eksik: {issues}')

    print(f'  R2_ENDPOINT_URL: {R2_ENDPOINT_URL}')
    print(f'  R2_BUCKET: {R2_BUCKET}')

    print('Startup: Model dosyaları indiriliyor...')
    download_from_r2('model/genre_vision_v3_best.pth', MODEL_PATH)
    download_from_r2('model/best_thresholds_v3.npy', THRESHOLDS_PATH)

    print('Startup: Model belleğe yükleniyor...')
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    label_cols = ckpt['label_cols']

    base = models.resnet18(weights=None)
    base.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(base.fc.in_features, len(label_cols))
    )
    base.load_state_dict(ckpt['model_state'])
    base.eval()
    model = base

    thresholds = np.load(THRESHOLDS_PATH)
    print(f'✓ Model yüklendi. {len(label_cols)} tür')
    print('=' * 60)


def run_inference(img: Image.Image) -> dict:
    """Bir PIL image'den tahmin döndürür. URL ve file endpoint'i ortak kullanır."""
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze().numpy()

    all_scores, confident_genres = {}, {}
    for i, tur in enumerate(label_cols):
        tur_adi = tur.replace('Tur_', '')
        yuzde = round(float(probs[i]) * 100, 2)
        all_scores[tur_adi] = yuzde
        if probs[i] > thresholds[i]:
            confident_genres[tur_adi] = yuzde

    sirali_all = dict(sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True))
    sirali_confident = dict(sorted(confident_genres.items(), key=lambda kv: kv[1], reverse=True))

    return {
        'turler': sirali_confident,
        'tum_skorlar': sirali_all,
        'top_5': dict(list(sirali_all.items())[:5])
    }


@app.get('/')
def root():
    return {
        'status': 'ok',
        'model': 'genre_vision_v3',
        'genres': len(label_cols) if label_cols else 0,
        'endpoints': ['POST /predict (JSON: {url})', 'POST /predict_file (multipart file)']
    }


@app.get('/health')
def health():
    return {'status': 'ok' if model is not None else 'loading'}


@app.post('/predict')
def predict_url(request: PredictionRequest):
    """URL'den afiş indir ve tahmin et"""
    if model is None:
        raise HTTPException(status_code=503, detail='Model henüz yüklenmedi')
    try:
        r = requests.get(request.url, timeout=8)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert('RGB')
        return run_inference(img)
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f'Görsel indirilemedi: {e}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Tahmin hatası: {e}')


@app.post('/predict_file')
async def predict_file(file: UploadFile = File(...)):
    """Yüklenen dosyadan tahmin et (frontend dosya yükleme akışı için)"""
    if model is None:
        raise HTTPException(status_code=503, detail='Model henüz yüklenmedi')

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400,
            detail=f'Görsel dosyası bekleniyor, alınan: {file.content_type}')

    try:
        contents = await file.read()
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail='Dosya 20 MB sınırını aşıyor')
        img = Image.open(BytesIO(contents)).convert('RGB')
        return run_inference(img)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Dosya işlenemedi: {e}')
