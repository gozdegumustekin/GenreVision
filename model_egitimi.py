import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# --- 1. PYTORCH DATASET SINIFI ---
class MoviePosterDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Hazırladığımız one-hot encoded CSV dosyasını okuyoruz
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # 'Tur_' ile başlayan tüm kolonları bul (Örn: Tur_Action, Tur_Drama)
        self.label_cols = [col for col in self.df.columns if col.startswith('Tur_')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Resim yolunu oluştur
        img_name = os.path.join(self.img_dir, f"{self.df.iloc[idx]['id']}.jpg")
        
        # Resmi aç ve RGB formatına çevir
        image = Image.open(img_name).convert('RGB')
        
        # Transform (Tensöre çevirme vb.) işlemlerini uygula
        if self.transform:
            image = self.transform(image)
            
        # Etiketleri (0 ve 1'leri) bir FloatTensöre çevir (BCEWithLogitsLoss float bekler)
        labels = self.df.iloc[idx][self.label_cols].values.astype('float32')
        labels = torch.tensor(labels)
        
        return image, labels

# --- 2. VERİ ÖN İŞLEME VE DATALOADER ---
# ResNet18'in ImageNet üzerinde eğitilirken kullandığı standart normalizasyon değerleri
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_yolu = './yapay_zeka_hazir_veri.csv'
afis_klasoru = './Afisler_224x224'

# Dataset nesnemizi oluşturuyoruz
dataset = MoviePosterDataset(csv_file=csv_yolu, img_dir=afis_klasoru, transform=transform)

# DataLoader: Verileri modele 32'şerli paketler (batch) halinde besleyecek
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Tür sayısını dinamik olarak bulalım
num_classes = len(dataset.label_cols)
print(f"Toplam tespit edilecek film türü sayısı: {num_classes}")

# --- 3. RESNET18 MİMARİSİ ---
# Pretrained (önceden eğitilmiş) modeli indiriyoruz
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ResNet18'in son katmanını (fc) bizim tür sayımıza eşitliyoruz
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

print("Model başarıyla oluşturuldu ve modifiye edildi!")