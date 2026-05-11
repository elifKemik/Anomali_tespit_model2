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

best.pt dosyasını buradan indirebilirsiniz: https://drive.google.com/file/d/17TRqony1SRjpeqG2v2eCjz-SmpKqgT45/view?usp=sharing

## 🚀 Kurulum ve Çalıştırma (Python 3.9 ile)

Bu proje Python 3.9 ile çalışmaktadır. Daha yüksek sürümlerde (3.10, 3.11, 3.12, 3.14) uyumluluk sorunu yaşanabilir.

Adım 1 - Python 3.9'u yükleyin
Python 3.9'u https://www.python.org/downloads/release/python-3913/ adresinden indirin. Kurulum sırasında "Add Python to PATH" seçeneğini işaretleyin.

Adım 2 - Projeyi klonlayın
git clone https://github.com/elifKemik/Anomali_tespit_model2
cd Anomali_Tespit_Modeli

Adım 3 - Sanal ortam oluşturun (Python 3.9 ile)
python -m venv venv
venv\Scripts\activate

Adım 4 - Gereksinimleri yükleyin
pip install -r requirements.txt

Adım 5 - best.pt dosyasını indirin
Google Drive linkinden best.pt dosyasını indirin ve proje klasörüne (Anomali_Tespit_Modeli) koyun.

Adım 6 - Uygulamayı başlatın
uvicorn web_app:app --reload

Adım 7 - Tarayıcıyı açın
http://127.0.0.1:8000 adresine gidin.

Not: Eğer "uvicorn bulunamadı" hatası alırsanız, önce pip install uvicorn komutunu çalıştırın.

## 📊 Tespit Edilen Anomaliler

0 - Arnold Chiari Malformation (Risk: Yüksek)
1 - Arachnoid Cyst (Risk: Orta)
2 - Cerebellar Hypoplasia (Risk: Yüksek)
3 - Cisterna Magna (Risk: Düşük)
4 - Colpocephaly (Risk: Orta)
5 - Encephalocele (Risk: Yüksek)
6 - Holoprosencephaly (Risk: Çok Yüksek)
7 - Hydranencephaly (Risk: Çok Yüksek)
8 - Intracranial Hemorrhage (Risk: Acil)
9 - Intracranial Tumor (Risk: Yüksek)
10 - Mild Ventriculomegaly (Risk: Düşük)
11 - Moderate Ventriculomegaly (Risk: Orta)
12 - Polencephaly (Risk: Yüksek)
13 - Severe Ventriculomegaly (Risk: Acil)

## 📁 Proje Yapısı

Anomali_Tespit_Modeli/ klasörü içinde web_app.py (web arayüzü), data.yaml (sınıf etiketleri), requirements.txt (gerekli kütüphaneler), best.pt (eğitilmiş model) ve static/ (CSS/JS dosyaları) bulunur.

## 📧 İletişim

GitHub: https://github.com/elifKemik