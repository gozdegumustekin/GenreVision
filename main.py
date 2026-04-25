from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GenreVision API", description="Film afişlerinden tür tahmini yapan yapay zeka servisi")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tüm web sitelerinden gelen isteklere izin ver
    allow_methods=["*"],
    allow_headers=["*"],
)

# Veri modelleri
class PredictionRequest(BaseModel):
    url: str

class PredictionResponse(BaseModel):
    turler: dict

# Global değişkenler
device = torch.device("cpu") # Sunucuda CPU kullanacağız
model = None
turler = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 
    'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.on_event("startup")
def load_model():
    global model
    print("Model belleğe yükleniyor...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(turler))
    # Colab'den indirdiğin pth dosyasının tam adını buraya yaz
    model.load_state_dict(torch.load('genre_vision_final.pth', map_location=device))
    model.eval()
    print("Model başarıyla yüklendi!")

@app.post("/predict", response_model=PredictionResponse)
def predict_genre(request: PredictionRequest):
    try:
        # Görseli indir
        response = requests.get(request.url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Ön işleme ve Tahmin
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.sigmoid(outputs).squeeze()
        
        # Sadece %10'dan büyük ihtimalleri JSON'a ekle
        sonuclar = {}
        for i, prob in enumerate(probabilities):
            yuzde = round(prob.item() * 100, 2)
            if yuzde > 10.0:
                sonuclar[turler[i]] = yuzde
                
        # Yüzdeye göre büyükten küçüğe sırala
        sirali_sonuclar = dict(sorted(sonuclar.items(), key=lambda item: item[1], reverse=True))
        
        return {"turler": sirali_sonuclar}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görsel işlenirken hata oluştu: {str(e)}")