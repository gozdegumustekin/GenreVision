import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
from tqdm import tqdm

# 1. Drive yerine doğrudan VS Code'da çalıştığın klasörü (ana dizini) kullanıyoruz
yol = './' 
print("Dosyalar okunuyor...")

# CSV'lerin kod dosyasıyla aynı klasörde olduğunu varsayıyoruz
df_m = pd.read_csv(yol + 'movies.csv')
df_g = pd.read_csv(yol + 'genres.csv')
df_p = pd.read_csv(yol + 'posters.csv')

# 2. Tabloları ID üzerinden birleştirelim
print("Tablolar birleştiriliyor...")
df = pd.merge(df_m, df_g, on='id')
df = pd.merge(df, df_p, on='id')

# 3. Rastgele 1000 örnek alalım (Pilot Veri)
df_pilot = df.sample(n=1000, random_state=42)

# 4. Kayıt klasörünü kendi bilgisayarında (yerel) hazırlayalım
cikti_yolu = yol + 'Afisler_224x224/'
if not os.path.exists(cikti_yolu):
    os.makedirs(cikti_yolu)

# 5. İndirme ve Boyutlandırma Döngüsü
print("\nİşlem başlıyor! 224x224 afişler yerel diskine kaydedilecek...")
hata_sayisi = 0

for index, satir in tqdm(df_pilot.iterrows(), total=1000):
    film_id = satir['id']
    url = satir['link']
    dosya_adi = f"{cikti_yolu}{film_id}.jpg"

    if os.path.exists(dosya_adi): continue # Varsa atla

    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert('RGB')
            img = img.resize((224, 224)) # Sihirli boyut!
            img.save(dosya_adi)
    except:
        hata_sayisi += 1

print(f"\nBitti! Hatalı link sayısı: {hata_sayisi}")
print(f"Klasörün burada: {os.path.abspath(cikti_yolu)}")

# --- 2. BÖLÜM: TEMİZLİK VE ENCODING ---

# 1. Kendi klasörümüze bakıp hangi resimlerin indiğini bulalım
inen_afisler = os.listdir(cikti_yolu)

# Sadece sonu .jpg olan dosyaların isimlerini (ID'lerini) alalım
basarili_idler = []
for dosya in inen_afisler:
    if dosya.endswith('.jpg'):
        film_id = int(dosya.replace('.jpg', ''))
        basarili_idler.append(film_id)

print(f"1. Klasörde bulunan sağlam afiş sayısı: {len(basarili_idler)}")

# 2. Sadece afişi olan filmleri tabloda bırakalım
df_temiz = df_pilot[df_pilot['id'].isin(basarili_idler)].copy()

# 3. Tür sütununu bulup 0 ve 1'lere çevirelim (One-Hot Encoding)
tur_sutunu = 'genre' if 'genre' in df_temiz.columns else 'tür'
print(f"2. '{tur_sutunu}' sütunu yapay zekanın anlayacağı 0 ve 1'lere çevriliyor...")

df_yapay_zeka = pd.get_dummies(df_temiz, columns=[tur_sutunu], prefix='Tur')

# True/False değerlerini sayısal 1 ve 0'a zorlayalım
bool_sutunlar = df_yapay_zeka.select_dtypes(include='bool').columns
df_yapay_zeka[bool_sutunlar] = df_yapay_zeka[bool_sutunlar].astype(int)

# 4. Bu mükemmel tabloyu yerel diskine kaydedelim
kayit_yolu = yol + 'yapay_zeka_hazir_veri.csv'
df_yapay_zeka.to_csv(kayit_yolu, index=False)

print("\nİşte Yapay Zekanın Göreceği Yeni Tablo (İlk 5 Satır):")
# VS Code (veya standart Python) display() fonksiyonunu tanımaz, print() kullanmalıyız.
print(df_yapay_zeka.head()) 
print(f"\n✅ MUHTEŞEM! Veri seti başarıyla bilgisayarına kaydedildi: {os.path.abspath(kayit_yolu)}")