FROM python:3.10-slim

WORKDIR /app

# İşletim sistemi seviyesinde gerekli olabilecek kütüphaneler
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# requirements.txt içindeki paketleri kur
RUN pip install --no-cache-dir -r requirements.txt

# Model dosyasını ve kodları kopyala
COPY . .

# FastAPI'ın varsayılan portunu aç
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]