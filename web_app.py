from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import base64
import uuid
import os
from pathlib import Path

app = FastAPI()

# Statik dosyalar için klasör oluştur
Path("static").mkdir(exist_ok=True)

# Modeli yükle
model = YOLO("best.pt")

# Kapsamlı anomali bilgileri
anomaly_info = {
    0: {
        'name': 'Arnold Chiari Malformation',
        'name_tr': 'Arnold Chiari Malformasyonu',
        'color': '#FF6B6B',
        'risk': 'Yüksek',
        'description': 'Beyincik bademciklerinin foramen magnumdan aşağıya doğru yer değiştirmesi durumudur. Beyin sapına bası yapabilir.',
        'symptoms': 'Baş ağrısı, boyun ağrısı, denge kaybı, yutma güçlüğü, uyku apnesi',
        'treatment': 'Cerrahi dekompresyon, semptomatik tedavi, fizik tedavi',
        'urgency': 'Erken teşhis önemlidir, nörolojik hasar kalıcı olabilir',
        'icon': '🧠'
    },
    1: {
        'name': 'Arachnoid Cyst',
        'name_tr': 'Araknoid Kist',
        'color': '#4ECDC4',
        'risk': 'Orta',
        'description': 'Beynin araknoid membranları arasında oluşan, beyin omurilik sıvısı ile dolu iyi huylu kistlerdir.',
        'symptoms': 'Çoğu asemptomatik, baş ağrısı, nöbet, hidrosefali',
        'treatment': 'Takip, semptomatikse cerrahi fenestrasyon veya şant',
        'urgency': 'Genellikle acil değildir, büyüme takibi yapılmalı',
        'icon': '💧'
    },
    2: {
        'name': 'Cerebellar Hypoplasia',
        'name_tr': 'Serebellar Hipoplazi',
        'color': '#45B7D1',
        'risk': 'Yüksek',
        'description': 'Beyinciğin normalden küçük veya az gelişmiş olması durumudur. Motor gelişimde sorunlara yol açar.',
        'symptoms': 'Motor gelişim geriliği, ataksi, hipotoni, nistagmus',
        'treatment': 'Destekleyici tedavi, fizik tedavi, ergoterapi',
        'urgency': 'Erken tanı ve rehabilitasyon önemlidir',
        'icon': '🎯'
    },
    3: {
        'name': 'Cisterna Magna',
        'name_tr': 'Sisterna Magna',
        'color': '#96CEB4',
        'risk': 'Düşük',
        'description': 'Beynin arka fossasında bulunan normal bir anatomik yapının genişlemesidir. Genellikle selimdir.',
        'symptoms': 'Genellikle asemptomatik, tesadüfen saptanır',
        'treatment': 'Tedavi gerektirmez, düzenli takip yeterlidir',
        'urgency': 'Normal varyant, acil değildir',
        'icon': '📐'
    },
    4: {
        'name': 'Colpocephaly',
        'name_tr': 'Kolposefali',
        'color': '#FFEAA7',
        'risk': 'Orta',
        'description': 'Lateral ventriküllerin arka boynuzlarının genişlemesi, ön boynuzların normal olması durumudur.',
        'symptoms': 'Nörogelişimsel gerilik, nöbet, motor bozukluklar',
        'treatment': 'Semptomatik tedavi, antiepileptikler, rehabilitasyon',
        'urgency': 'Nörolojik takip gereklidir',
        'icon': '📏'
    },
    5: {
        'name': 'Encephalocele',
        'name_tr': 'Ensefalosel',
        'color': '#DDA0DD',
        'risk': 'Yüksek',
        'description': 'Kafatası defektinden beyin dokusu ve meninkslerin dışarıya herniasyonudur.',
        'symptoms': 'Kafada şişlik, nörolojik defisit, mikrosefali, gelişim geriliği',
        'treatment': 'Cerrahi onarım, nörolojik rehabilitasyon',
        'urgency': 'ACİL - Erken cerrahi müdahale gerekir',
        'icon': '🔴'
    },
    6: {
        'name': 'Holoprosencephaly',
        'name_tr': 'Holoprozensefali',
        'color': '#FF9999',
        'risk': 'Çok Yüksek',
        'description': 'Ön beynin tam veya kısmi ayrılmaması ile karakterize ciddi bir beyin malformasyonudur.',
        'symptoms': 'Yüz anomalileri, nöbet, gelişim geriliği, hipotoni',
        'treatment': 'Destekleyici tedavi, antikonvülzanlar, palyatif bakım',
        'urgency': 'ACİL - İleri düzey nörolojik takip şart',
        'icon': '⚠️'
    },
    7: {
        'name': 'Hydranencephaly',
        'name_tr': 'Hidranensefali',
        'color': '#B19CD9',
        'risk': 'Çok Yüksek',
        'description': 'Beyin hemisferlerinin yok olduğu, yerini beyin omurilik sıvısının doldurduğu ciddi durum.',
        'symptoms': 'Makrosefali, beslenme güçlüğü, nöbet, gelişim geriliği',
        'treatment': 'Palyatif bakım, ventriküloperitoneal şant',
        'urgency': 'ACİL - Yoğun tıbbi bakım gereklidir',
        'icon': '🚨'
    },
    8: {
        'name': 'Intracranial Hemorrhage',
        'name_tr': 'İntrakranyal Kanama',
        'color': '#FF6666',
        'risk': 'Acil',
        'description': 'Kafatası içinde kanama varlığıdır. Çeşitli derecelerde olabilir.',
        'symptoms': 'Baş ağrısı, bilinç değişikliği, nöbet, fokal nörolojik defisit',
        'treatment': 'Hastaneye yatış, antikonvülzanlar, gerekirse cerrahi',
        'urgency': 'ACİL - Acil tıbbi müdahale gerekir!',
        'icon': '🩸'
    },
    9: {
        'name': 'Intracranial Tumor',
        'name_tr': 'İntrakranyal Tümör',
        'color': '#FFB347',
        'risk': 'Yüksek',
        'description': 'Beyin içinde veya çevresinde kitlesel oluşumdur. İyi veya kötü huylu olabilir.',
        'symptoms': 'Baş ağrısı, bulantı, kusma, nöbet, fokal nörolojik defisit',
        'treatment': 'Cerraji, radyoterapi, kemoterapi, takip',
        'urgency': 'İleri tetkik ve onkoloji konsültasyonu gerekir',
        'icon': '🎗️'
    },
    10: {
        'name': 'Mild Ventriculomegaly',
        'name_tr': 'Hafif Ventrikülomegali',
        'color': '#77DD77',
        'risk': 'Düşük',
        'description': 'Beyin ventriküllerinin hafif derecede genişlemesidir. 10-12mm arası.',
        'symptoms': 'Çoğunlukla asemptomatik, takip gerektirir',
        'treatment': 'Düzenli ultrason takibi, nörogelişimsel değerlendirme',
        'urgency': 'Düzenli takip yeterlidir',
        'icon': '📊'
    },
    11: {
        'name': 'Moderate Ventriculomegaly',
        'name_tr': 'Orta Ventrikülomegali',
        'color': '#FFD700',
        'risk': 'Orta',
        'description': 'Ventriküllerin orta derecede genişlemesi. 12-15mm arası.',
        'symptoms': 'Baş çevresinde artış, gelişim geriliği olabilir',
        'treatment': 'Nöroloji takibi, ultrason takibi, gerekirse MR',
        'urgency': 'Nöroloji konsültasyonu önerilir',
        'icon': '📈'
    },
    12: {
        'name': 'Polencephaly',
        'name_tr': 'Polensefali',
        'color': '#CF9FFF',
        'risk': 'Yüksek',
        'description': 'Beyin korteksinde çok sayıda kist benzeri boşlukların olduğu nadir bir malformasyon.',
        'symptoms': 'Nöbet, gelişim geriliği, spastisite, beslenme güçlüğü',
        'treatment': 'Antiepileptikler, palyatif bakım, rehabilitasyon',
        'urgency': 'Nörolojik takip gerektirir',
        'icon': '🌀'
    },
    13: {
        'name': 'Severe Ventriculomegaly',
        'name_tr': 'Şiddetli Ventrikülomegali',
        'color': '#FF4B4B',
        'risk': 'Acil',
        'description': 'Ventriküllerin >15mm üzerinde genişlemesi, hidrosefali bulguları olabilir.',
        'symptoms': 'Makrosefali, fontanel gerginliği, kusma, beslenme sorunları',
        'treatment': 'Ventriküloperitoneal şant, endoskopik üçüncü ventrikülostomi',
        'urgency': 'ACİL - Beyin cerrahi değerlendirmesi gerekir!',
        'icon': '🔴'
    }
}


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ultrason Anomali Tespit Sistemi | Klinik Bilgi Destekli Tanı</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }

            /* Header */
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }

            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }

            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }

            .stats {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 20px;
                flex-wrap: wrap;
            }

            .stat-card {
                background: rgba(255,255,255,0.2);
                padding: 10px 20px;
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }

            /* Upload Area */
            .upload-area {
                padding: 40px;
                text-align: center;
                background: #f8f9fa;
                border-bottom: 1px solid #e0e0e0;
            }

            .upload-box {
                border: 3px dashed #667eea;
                border-radius: 20px;
                padding: 40px;
                background: white;
                cursor: pointer;
                transition: all 0.3s;
            }

            .upload-box:hover {
                border-color: #764ba2;
                background: #faf5ff;
                transform: scale(1.02);
            }

            .upload-icon {
                font-size: 48px;
                margin-bottom: 10px;
            }

            .analyze-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 30px;
                font-size: 18px;
                cursor: pointer;
                margin-top: 20px;
                transition: all 0.3s;
            }

            .analyze-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102,126,234,0.4);
            }

            .analyze-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            /* Loading */
            .loading {
                text-align: center;
                padding: 40px;
                display: none;
            }

            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            /* Results Area */
            .results-area {
                padding: 40px;
                display: none;
            }

            .diagnosis-header {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
            }

            .verdict {
                font-size: 2em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            .verdict.anomaly {
                color: #dc3545;
            }

            .verdict.normal {
                color: #28a745;
            }

            .confidence-bar {
                background: #e0e0e0;
                border-radius: 10px;
                overflow: hidden;
                margin: 20px 0;
            }

            .confidence-fill {
                background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
                height: 40px;
                border-radius: 10px;
                transition: width 0.5s;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 1.1em;
            }

            /* Image Comparison */
            .image-comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }

            .image-card {
                background: #f8f9fa;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                transition: transform 0.3s;
            }

            .image-card:hover {
                transform: translateY(-5px);
            }

            .image-card h3 {
                background: #667eea;
                color: white;
                padding: 15px;
                margin: 0;
            }

            .image-card img {
                width: 100%;
                height: auto;
                display: block;
            }

            /* Disease Info Card */
            .disease-info {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 25px;
                margin: 30px 0;
                display: none;
            }

            .disease-header {
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }

            .disease-icon {
                font-size: 48px;
            }

            .disease-title {
                flex: 1;
            }

            .disease-title h2 {
                color: #333;
                margin-bottom: 5px;
            }

            .disease-badge {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 0.85em;
            }

            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }

            .info-card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: all 0.3s;
            }

            .info-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            }

            .info-card h4 {
                color: #667eea;
                margin-bottom: 12px;
                font-size: 1.2em;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .info-card p {
                color: #555;
                line-height: 1.6;
            }

            .info-card ul {
                list-style: none;
                padding: 0;
            }

            .info-card li {
                padding: 5px 0;
                color: #555;
                border-bottom: 1px solid #f0f0f0;
            }

            .info-card li:last-child {
                border-bottom: none;
            }

            /* Detections Table */
            .detections-section {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                margin-top: 30px;
            }

            .detections-section h3 {
                margin-bottom: 20px;
                color: #333;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 10px;
                overflow: hidden;
            }

            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }

            th {
                background: #667eea;
                color: white;
                cursor: pointer;
            }

            th:hover {
                background: #764ba2;
            }

            .risk-badge {
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: bold;
                display: inline-block;
            }

            .risk-yuksek { background: #ff6b6b; color: white; }
            .risk-orta { background: #ffd93d; color: #333; }
            .risk-dusuk { background: #6bcb77; color: white; }
            .risk-acil { background: #dc3545; color: white; animation: pulse 1s infinite; }
            .risk-cok-yuksek { background: #8b0000; color: white; }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }

            .treatment-card {
                background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                border-left: 5px solid #667eea;
            }

            /* Footer */
            .footer {
                background: #333;
                color: white;
                text-align: center;
                padding: 20px;
                font-size: 0.9em;
            }

            /* Responsive */
            @media (max-width: 768px) {
                .image-comparison {
                    grid-template-columns: 1fr;
                }
                .stats {
                    flex-direction: column;
                }
                .info-grid {
                    grid-template-columns: 1fr;
                }
                .header h1 {
                    font-size: 1.5em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 Ultrason Anomali Tespit Sistemi</h1>
                <p>Yapay Zeka Destekli Klinik Karar Destek Sistemi | YOLOv8 ile Gerçek Zamanlı Analiz</p>
                <div class="stats">
                    <div class="stat-card">🎯 14 Farklı Anomali Tipi</div>
                    <div class="stat-card">⚡ %91 Doğruluk Oranı</div>
                    <div class="stat-card">📚 Klinik Bilgi Desteği</div>
                    <div class="stat-card">🩺 Tedavi Önerileri</div>
                </div>
            </div>

            <div class="upload-area">
                <div class="upload-box" onclick="document.getElementById('imageInput').click()">
                    <div class="upload-icon">📸</div>
                    <div>Ultrason Görüntüsü Seçmek İçin Tıklayın</div>
                    <small style="color: #666;">Desteklenen formatlar: JPG, PNG, JPEG</small>
                    <input type="file" id="imageInput" accept="image/*" style="display: none;" onchange="previewImage()">
                </div>
                <div id="fileName" style="margin-top: 10px; color: #666;"></div>
                <button class="analyze-btn" id="analyzeBtn" onclick="analyze()">🔍 Klinik Analizi Başlat</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Yapay Zeka Görüntüyü Analiz Ediyor ve Klinik Bilgileri Derliyor...</p>
                <small>Bu işlem 2-3 saniye sürebilir</small>
            </div>

            <div class="results-area" id="resultsArea">
                <div class="diagnosis-header" id="diagnosisHeader"></div>

                <div class="image-comparison">
                    <div class="image-card">
                        <h3>📷 Yüklenen Ultrason Görüntüsü</h3>
                        <img id="originalImage" alt="Orijinal görüntü">
                    </div>
                    <div class="image-card">
                        <h3>🎯 Yapay Zeka Tespiti (İşaretli Alanlar)</h3>
                        <img id="annotatedImage" alt="İşaretli görüntü">
                    </div>
                </div>

                <div id="clinicalInfo" class="disease-info"></div>

                <div class="detections-section" id="detectionsSection" style="display: none;">
                    <h3>📊 Tespit Edilen Anomalilerin Detaylı Klinik Analizi</h3>
                    <table id="detectionsTable">
                        <thead>
                            <tr><th>No</th><th>Anomali Tipi</th><th>Risk Seviyesi</th><th>Doğruluk Oranı</th><th>Değerlendirme</th></tr>
                        </thead>
                        <tbody id="tableBody"></tbody>
                    </table>
                </div>
            </div>

            <div class="footer">
                🧠 Yapay Zeka Destekli Klinik Karar Destek Sistemi | Bu sistem tanı koymaz, klinik değerlendirme için yardımcı bilgiler sunar.
            </div>
        </div>

        <script>
            let currentFile = null;
            let anomalyData = null;

            function previewImage() {
                const file = document.getElementById('imageInput').files[0];
                if (file) {
                    currentFile = file;
                    document.getElementById('fileName').innerHTML = `📎 Seçilen dosya: ${file.name}`;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('originalImage').src = e.target.result;
                    };
                    reader.readAsDataURL(file);
                }
            }

            async function analyze() {
                if (!currentFile) {
                    alert('Lütfen önce bir ultrason görüntüsü seçin!');
                    return;
                }

                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultsArea').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = true;

                const formData = new FormData();
                formData.append('file', currentFile);

                try {
                    const response = await fetch('/predict/', { method: 'POST', body: formData });
                    const data = await response.json();

                    // Diagnosis Header
                    const header = document.getElementById('diagnosisHeader');
                    if (data.anomaly_detected && data.detections.length > 0) {
                        header.innerHTML = `
                            <div class="verdict anomaly">
                                ⚠️ ANOMALİ TESPİT EDİLDİ - KLİNİK DEĞERLENDİRME GEREKLİ
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${data.detections[0].confidence}%">
                                    🎯 Yapay Zeka Güven Oranı: %${data.detections[0].confidence}
                                </div>
                            </div>
                        `;
                    } else {
                        header.innerHTML = `
                            <div class="verdict normal">
                                ✅ NORMAL - ANOMALİ TESPİT EDİLMEDİ
                            </div>
                            <div style="text-align: center; margin-top: 20px; padding: 20px;">
                                <p>🎉 Ultrason görüntüsü normal anatomik yapılar göstermektedir.</p>
                                <p>Düzenli takip ve kontrol önerilir.</p>
                            </div>
                        `;
                    }

                    // Annotated Image
                    document.getElementById('annotatedImage').src = `data:image/png;base64,${data.annotated_image}`;

                    // Clinical Info
                    if (data.detections.length > 0) {
                        anomalyData = data.detections;
                        displayClinicalInfo(anomalyData);
                    } else {
                        document.getElementById('clinicalInfo').style.display = 'none';
                    }

                    // Detections Table
                    if (data.detections.length > 0) {
                        document.getElementById('detectionsSection').style.display = 'block';
                        const tableBody = document.getElementById('tableBody');
                        tableBody.innerHTML = '';

                        data.detections.forEach((det, index) => {
                            const riskClass = det.risk === 'Acil' ? 'risk-acil' : 
                                             det.risk === 'Çok Yüksek' ? 'risk-cok-yuksek' :
                                             det.risk === 'Yüksek' ? 'risk-yuksek' :
                                             det.risk === 'Orta' ? 'risk-orta' : 'risk-dusuk';

                            const riskTurkish = det.risk === 'Acil' ? '🔴 ACİL' :
                                               det.risk === 'Çok Yüksek' ? '⚠️ Çok Yüksek' :
                                               det.risk === 'Yüksek' ? '🟠 Yüksek' :
                                               det.risk === 'Orta' ? '🟡 Orta' : '🟢 Düşük';

                            const evaluation = det.confidence > 80 ? '🎯 Yüksek olasılık - Klinik korelasyon önerilir' :
                                              det.confidence > 50 ? '⚠️ Orta olasılık - İleri tetkik gerekli' :
                                              '🔍 Düşük olasılık - Tekrar değerlendirin';

                            const row = tableBody.insertRow();
                            row.insertCell(0).innerHTML = index + 1;
                            row.insertCell(1).innerHTML = `<strong>${det.class}</strong>`;
                            row.insertCell(2).innerHTML = `<span class="risk-badge ${riskClass}">${riskTurkish}</span>`;
                            row.insertCell(3).innerHTML = `<span style="color: ${det.confidence > 80 ? '#28a745' : det.confidence > 50 ? '#ffc107' : '#dc3545'}; font-weight: bold;">%${det.confidence}</span>`;
                            row.insertCell(4).innerHTML = evaluation;

                            row.style.cursor = 'pointer';
                            row.onclick = () => displaySingleDiseaseInfo(det);
                        });
                    } else {
                        document.getElementById('detectionsSection').style.display = 'none';
                    }

                    document.getElementById('resultsArea').style.display = 'block';
                } catch (error) {
                    alert('Analiz sırasında bir hata oluştu: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                }
            }

            function displayClinicalInfo(detections) {
                if (!detections || detections.length === 0) return;

                let html = '';
                detections.forEach((det, index) => {
                    const riskClass = det.risk === 'Acil' ? 'risk-acil' : 
                                     det.risk === 'Çok Yüksek' ? 'risk-cok-yuksek' :
                                     det.risk === 'Yüksek' ? 'risk-yuksek' : 'risk-orta';

                    html += `
                        <div style="margin-bottom: ${index < detections.length-1 ? '30px' : '0'}; padding: ${index < detections.length-1 ? '0 0 30px 0' : '0'}; border-bottom: ${index < detections.length-1 ? '2px solid #e0e0e0' : 'none'}">
                            <div class="disease-header">
                                <div class="disease-icon">${det.icon || '🏥'}</div>
                                <div class="disease-title">
                                    <h2>${det.class}</h2>
                                    <div>
                                        <span class="disease-badge ${riskClass}" style="margin-right: 10px;">Risk: ${det.risk}</span>
                                        <span class="disease-badge" style="background: #667eea; color: white;">Doğruluk: %${det.confidence}</span>
                                    </div>
                                </div>
                            </div>
                            <div class="info-grid">
                                <div class="info-card">
                                    <h4>📖 Hastalık Tanımı</h4>
                                    <p>${det.description || 'Klinik bilgi bulunamadı.'}</p>
                                </div>
                                <div class="info-card">
                                    <h4>⚠️ Belirtiler</h4>
                                    <ul>
                                        ${det.symptoms ? det.symptoms.split(',').map(s => `<li>• ${s.trim()}</li>`).join('') : '<li>• Bilgi bulunamadı</li>'}
                                    </ul>
                                </div>
                                <div class="info-card treatment-card">
                                    <h4>💊 Tedavi Yaklaşımı</h4>
                                    <p>${det.treatment || 'Tedavi bilgisi bulunamadı. Klinik değerlendirme önerilir.'}</p>
                                </div>
                                <div class="info-card">
                                    <h4>🚨 Aciliyet Değerlendirmesi</h4>
                                    <p><strong>${det.urgency || 'Değerlendirme gerekli'}</strong></p>
                                </div>
                            </div>
                        </div>
                    `;
                });

                document.getElementById('clinicalInfo').innerHTML = html;
                document.getElementById('clinicalInfo').style.display = 'block';
            }

            function displaySingleDiseaseInfo(det) {
                const riskClass = det.risk === 'Acil' ? 'risk-acil' : 
                                 det.risk === 'Çok Yüksek' ? 'risk-cok-yuksek' :
                                 det.risk === 'Yüksek' ? 'risk-yuksek' : 'risk-orta';

                document.getElementById('clinicalInfo').innerHTML = `
                    <div class="disease-header">
                        <div class="disease-icon">${det.icon || '🏥'}</div>
                        <div class="disease-title">
                            <h2>${det.class}</h2>
                            <div>
                                <span class="disease-badge ${riskClass}" style="margin-right: 10px;">Risk: ${det.risk}</span>
                                <span class="disease-badge" style="background: #667eea; color: white;">Doğruluk: %${det.confidence}</span>
                            </div>
                        </div>
                    </div>
                    <div class="info-grid">
                        <div class="info-card">
                            <h4>📖 Hastalık Tanımı</h4>
                            <p>${det.description || 'Klinik bilgi bulunamadı.'}</p>
                        </div>
                        <div class="info-card">
                            <h4>⚠️ Belirtiler</h4>
                            <ul>
                                ${det.symptoms ? det.symptoms.split(',').map(s => `<li>• ${s.trim()}</li>`).join('') : '<li>• Bilgi bulunamadı</li>'}
                            </ul>
                        </div>
                        <div class="info-card treatment-card">
                            <h4>💊 Tedavi Yaklaşımı</h4>
                            <p>${det.treatment || 'Tedavi bilgisi bulunamadı. Klinik değerlendirme önerilir.'}</p>
                        </div>
                        <div class="info-card">
                            <h4>🚨 Aciliyet Değerlendirmesi</h4>
                            <p><strong>${det.urgency || 'Değerlendirme gerekli'}</strong></p>
                        </div>
                    </div>
                `;
                document.getElementById('clinicalInfo').style.display = 'block';

                // Kaydır
                document.getElementById('clinicalInfo').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        </script>
    </body>
    </html>
    """


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    results = model(temp_path)
    result = results[0]

    boxes = result.boxes
    detections = []

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            info = anomaly_info.get(cls_id, {
                'name': f"Anomali Tipi {cls_id}",
                'name_tr': 'Bilinmeyen',
                'color': '#999',
                'risk': 'Bilinmiyor',
                'description': 'Bu anomali hakkında detaylı klinik bilgi bulunmamaktadır.',
                'symptoms': 'Değerlendirme gerekli',
                'treatment': 'Klinik değerlendirme önerilir',
                'urgency': 'Uzman değerlendirmesi gerekli',
                'icon': '🏥'
            })
            detections.append({
                "class": info['name_tr'],
                "class_en": info['name'],
                "confidence": round(conf, 2),
                "risk": info['risk'],
                "description": info['description'],
                "symptoms": info['symptoms'],
                "treatment": info['treatment'],
                "urgency": info['urgency'],
                "icon": info['icon']
            })

    annotated_img = result.plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', annotated_img_rgb)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    os.remove(temp_path)

    return {
        "anomaly_detected": len(detections) > 0,
        "detections": detections,
        "annotated_image": img_base64
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)