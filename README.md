# 🩺 Ultrason Görüntüsü Anomali Tespit Sistemi

Yapay zeka (YOLOv8) kullanarak ultrason görüntülerinde fetal anomali tespiti yapan web tabanlı sistem.

## 📋 Özellikler

- 🎯 14 farklı anomali tipi tespiti (Arnold Chiari, Encephalocele, Holoprosencephaly, Ventrikülomegali vb.)
- ⚡ Gerçek zamanlı analiz (2-3 saniye)
- 🌐 Modern ve estetik web arayüzü
- 📊 Görsel işaretleme (bounding box ile anomali bölgesi)
- 📈 Doğruluk oranı göstergesi (%0-100)
- ⚠️ Risk seviyesi değerlendirmesi (Düşük/Orta/Yüksek/Acil)
- 📖 Hastalık tanımı, belirtiler ve tedavi önerileri
- 🖼️ Öncesi/sonrası görüntü karşılaştırma
- 📱 Mobil uyumlu tasarım

## 📥 Model Dosyasını İndir

best.pt dosyasını buradan indirebilirsiniz: https://drive.google.com/file/d/1lP-szJZvr0uo-mHwuKQqolsridkPutyc/view?usp=sharing

## 📈 Model Performansı

| Metrik | Değer |
|--------|-------|
| mAP50 | %95.5 |
| Ortalama Precision | %92.4 |
| Ortalama Recall | %91.8 |

## 🚀 Kurulum ve Çalıştırma 

Adım 1 - Projeyi klonlayın
git clone https://github.com/elifKemik/Anomali_tespit_model2
cd Anomali_Tespit_Modeli

Adım 2 - Sanal ortam oluşturun 
python -m venv venv
venv\Scripts\activate

Adım 3 - Gereksinimleri yükleyin
pip install -r requirements.txt

Adım 4 - best.pt dosyasını indirin
Google Drive linkinden best.pt dosyasını indirin ve proje klasörüne (Anomali_Tespit_Modeli) koyun.

Adım 5 - Uygulamayı başlatın
uvicorn web_app:app --reload

Adım 6 - Tarayıcıyı açın
http://127.0.0.1:8000 adresine gidin.

Not: Eğer "uvicorn bulunamadı" hatası alırsanız, önce pip install uvicorn komutunu çalıştırın.

## 📊 Tespit Edilen Anomaliler

| ID | Anomali Tipi | Risk Seviyesi |
|----|--------------|---------------|
| 0 | Arnold Chiari Malformation | Yüksek |
| 1 | Arachnoid Cyst | Orta |
| 2 | Cerebellar Hypoplasia | Yüksek |
| 3 | Cisterna Magna | Düşük |
| 4 | Colpocephaly | Orta |
| 5 | Encephalocele | Yüksek |
| 6 | Holoprosencephaly | Çok Yüksek |
| 7 | Hydranencephaly | Çok Yüksek |
| 8 | Intracranial Hemorrhage | Acil |
| 9 | Intracranial Tumor | Yüksek |
| 10 | Mild Ventriculomegaly | Düşük |
| 11 | Moderate Ventriculomegaly | Orta |
| 12 | Polencephaly | Yüksek |
| 13 | Severe Ventriculomegaly | Acil |


## 📁 Proje Yapısı

Anomali_Tespit_Modeli/ klasörü içinde web_app.py (web arayüzü), data.yaml (sınıf etiketleri), requirements.txt (gerekli kütüphaneler) ve best.pt (eğitilmiş model) bulunur.

## 📧 İletişim

GitHub: https://github.com/elifKemik