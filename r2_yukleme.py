import os
import boto3
from botocore.exceptions import NoCredentialsError
from tqdm import tqdm

# --- 1. CLOUDFLARE R2 BİLGİLERİNİ BURAYA GİR ---
ACCESS_KEY = '26ff1aa1645b69cac212507ec4c079dd'
SECRET_KEY = '37872cda19b45e33509d5b84a3514234a5a06755481363e1561cb7a404f23a77'
ENDPOINT_URL = 'https://8a97a7792885ace66acb04fa79b9ded7.r2.cloudflarestorage.com' # Örn: https://<hesap-id>.r2.cloudflarestorage.com
BUCKET_NAME = 'genrevision' 

# --- 2. BOTO3 İSTEMCİSİNİ BAŞLAT ---
# R2, AWS S3 mimarisi ile çalıştığı için 's3' servisini kullanıyoruz
s3 = boto3.client(
    service_name='s3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='auto' # Cloudflare R2 için region her zaman 'auto' kalmalıdır
)

# --- 3. YÜKLENECEK KLASÖR VE DOSYALAR ---
klasor_yolu = './Afisler_224x224'

# Klasörün var olup olmadığını kontrol et
if not os.path.exists(klasor_yolu):
    print(f"Hata: '{klasor_yolu}' klasörü bulunamadı. Lütfen önce veri hazırlık kodunu çalıştırdığından emin ol.")
    exit()

# Sadece .jpg uzantılı dosyaları bul
yuklenecek_dosyalar = [f for f in os.listdir(klasor_yolu) if f.endswith('.jpg')]

if len(yuklenecek_dosyalar) == 0:
    print("Klasörde yüklenecek .jpg dosyası bulunamadı.")
    exit()

print(f"Toplam {len(yuklenecek_dosyalar)} afiş R2 bucket'ına ('{BUCKET_NAME}') yükleniyor...")

# --- 4. YÜKLEME DÖNGÜSÜ ---
basarili_sayisi = 0
hata_sayisi = 0

for dosya in tqdm(yuklenecek_dosyalar, desc="Yükleniyor"):
    dosya_tam_yolu = os.path.join(klasor_yolu, dosya)
    try:
        # s3.upload_file(yerel_dosya_yolu, bucket_adi, r2_icindeki_dosya_adi)
        s3.upload_file(dosya_tam_yolu, BUCKET_NAME, dosya)
        basarili_sayisi += 1
    except NoCredentialsError:
        print("\nHata: R2 kimlik bilgileri (Access Key / Secret Key) geçersiz veya eksik!")
        break
    except Exception as e:
        print(f"\nHata ({dosya} yüklenirken): {e}")
        hata_sayisi += 1

print("\n--- YÜKLEME ÖZETİ ---")
print(f"Başarıyla Yüklenen: {basarili_sayisi}")
print(f"Hatalı: {hata_sayisi}")