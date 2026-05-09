import os
import numpy as np
import torch
import torch.nn as nn
import requests
import boto3
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import models, transforms
from PIL import Image

# === KONFIGÜRASYON (Railway environment variables'tan okunur) ===
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY')
R2_SECRET_KEY = os.getenv('R2_SECRET_KEY')
R2_ENDPOINT_URL = os.getenv('R2_ENDPOINT_URL')
R2_BUCKET = os.getenv('R2_BUCKET', 'genrevision')

MODEL_PATH = 'genre_vision_v3_best.pth'
THRESHOLDS_PATH = 'best_thresholds_v3.npy'

app = FastAPI(title='GenreVision API v3',
              description='Film afişlerinden tür tahmini (ResNet18 + threshold tuning)')

app.add_middleware(CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# === GLOBAL ===
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


def download_from_r2(key: str, local_path: str):
    """R2'den dosya indir (sadece yoksa)"""
    if os.path.exists(local_path):
        print(f'  {local_path} zaten var, atlandı')
        return
    s3 = boto3.client('s3', endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY,
        region_name='auto')
    s3.download_file(R2_BUCKET, key, local_path)
    print(f'  ✓ {key} indirildi → {local_path}')


@app.on_event('startup')
def load_model():
    global model, label_cols, thresholds
    print('Startup: Model dosyaları indiriliyor...')

    download_from_r2('model/genre_vision_v3_best.pth', MODEL_PATH)
    download_from_r2('model/best_thresholds_v3.npy', THRESHOLDS_PATH)

    print('Startup: Model belleğe yükleniyor...')
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    label_cols = ckpt['label_cols']

    # v3 mimarisi: Dropout + Linear (eğitimle birebir aynı olmalı)
    base = models.resnet18(weights=None)
    base.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(base.fc.in_features, len(label_cols))
    )
    base.load_state_dict(ckpt['model_state'])
    base.eval()
    model = base

    thresholds = np.load(THRESHOLDS_PATH)
    print(f'✓ Model yüklendi. {len(label_cols)} tür, '
          f'epoch {ckpt.get("epoch", "?")}, '
          f'val F1_macro {ckpt.get("val_f1_macro", 0):.3f}')


@app.get('/')
def root():
    return {'status': 'ok', 'model': 'genre_vision_v3', 'genres': len(label_cols) if label_cols else 0}


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/predict')
def predict_genre(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='Model henüz yüklenmedi')

    try:
        # 1. Görseli indir
        r = requests.get(request.url, timeout=8)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert('RGB')

        # 2. Ön işleme + tahmin
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).squeeze().numpy()

        # 3. İki çıktı:
        #    a) Eşiği geçen türler (model'in "evet" dediği)
        #    b) Tüm türler için olasılıklar (UI tablosu için)
        all_scores = {}
        confident_genres = {}
        for i, tur in enumerate(label_cols):
            tur_adi = tur.replace('Tur_', '')
            yuzde = round(float(probs[i]) * 100, 2)
            all_scores[tur_adi] = yuzde
            if probs[i] > thresholds[i]:
                confident_genres[tur_adi] = yuzde

        # Büyükten küçüğe sırala
        sirali_all = dict(sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True))
        sirali_confident = dict(sorted(confident_genres.items(), key=lambda kv: kv[1], reverse=True))

        return {
            'turler': sirali_confident,           # Frontend bunu kullanır (genre bar listesi)
            'tum_skorlar': sirali_all,            # Debug için tam skor tablosu
            'top_5': dict(list(sirali_all.items())[:5])  # Her zaman top-5
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f'Görsel indirilemedi: {e}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Tahmin hatası: {e}')