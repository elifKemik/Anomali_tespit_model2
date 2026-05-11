import os
from pathlib import Path

# ==================== 1. VERİ SETİNİ TARA ====================
# Kendi veri setinin yolunu yaz
veri_seti_yolu = "C:/Users/eserr/Desktop/egitim/dataset"

# ==================== 2. FONKSİYON: Bir klasördeki normal/anomali sayısını bul ====================
def klasoru_tara(images_folder):
    """
    images klasöründeki her resme karşılık gelen etiket dosyasını okur
    ve normal/anomali sayısını döndürür.
    """
    normal_sayisi = 0
    anomali_sayisi = 0
    toplam_resim = 0
    
    # Etiketlerin olduğu klasör (images yerine labels)
    labels_folder = images_folder.parent / "labels"
    
    print(f"   Aranan etiket klasörü: {labels_folder}")
    
    if not labels_folder.exists():
        print(f"   ❌ Etiket klasörü bulunamadı: {labels_folder}")
        return 0, 0, 0
    
    # Tüm resimleri tara (.jpg, .png, .jpeg)
    resim_uzantilari = ["*.jpg", "*.png", "*.jpeg"]
    for uzanti in resim_uzantilari:
        for img_path in images_folder.glob(uzanti):
            toplam_resim += 1
            
            # Aynı isimli etiket dosyasını bul (.txt)
            label_path = labels_folder / (img_path.stem + ".txt")
            
            if not label_path.exists():
                print(f"   Uyarı: Etiket dosyası bulunamadı: {label_path}")
                continue
            
            # Etiket dosyasını oku
            try:
                with open(label_path, 'r') as f:
                    ilk_satir = f.readline().strip()
                    if ilk_satir:
                        parts = ilk_satir.split()
                        if len(parts) > 0:
                            class_id = int(parts[0])
                        else:
                            continue
                    else:
                        continue  # Boş dosya
                
                # Normal sınıf kontrolü (0 numaralı sınıf background/normal ise)
                if class_id == 0:
                    normal_sayisi += 1
                else:
                    anomali_sayisi += 1
                    
            except Exception as e:
                print(f"   Hata okurken {label_path}: {e}")
                continue
    
    return normal_sayisi, anomali_sayisi, toplam_resim

# ==================== 3. EĞİTİM VE VALİDASYON ORANLARINI HESAPLA ====================
print("\n" + "="*50)
print("VERİ SETİ ANALİZİ")
print("="*50)

# Train klasörü
train_images_folder = Path(veri_seti_yolu) / "train" / "images"
print(f"\n📁 Train images klasörü: {train_images_folder}")

if train_images_folder.exists():
    print("   🔍 Taranıyor...")
    train_normal, train_anomali, train_toplam = klasoru_tara(train_images_folder)
    
    if train_toplam > 0:
        train_normal_ratio = train_normal / train_toplam
        print(f"\n📊 Eğitim Seti:")
        print(f"   Toplam: {train_toplam} resim")
        print(f"   Sağlıklı (normal): {train_normal} (%{train_normal_ratio*100:.1f})")
        print(f"   Anomalili: {train_anomali} (%{(1-train_normal_ratio)*100:.1f})")
    else:
        print("   ⚠️ Hiç resim bulunamadı!")
else:
    print(f"   ❌ Train klasörü bulunamadı: {train_images_folder}")

# Valid klasörü
val_images_folder = Path(veri_seti_yolu) / "valid" / "images"  # DİKKAT: "val" değil "valid"
print(f"\n📁 Valid images klasörü: {val_images_folder}")

if val_images_folder.exists():
    print("   🔍 Taranıyor...")
    val_normal, val_anomali, val_toplam = klasoru_tara(val_images_folder)
    
    if val_toplam > 0:
        val_normal_ratio = val_normal / val_toplam
        print(f"\n📊 Validation Seti:")
        print(f"   Toplam: {val_toplam} resim")
        print(f"   Sağlıklı (normal): {val_normal} (%{val_normal_ratio*100:.1f})")
        print(f"   Anomalili: {val_anomali} (%{(1-val_normal_ratio)*100:.1f})")
    else:
        print("   ⚠️ Hiç resim bulunamadı!")
else:
    # Alternatif: klasör "val" olabilir
    val_images_folder = Path(veri_seti_yolu) / "val" / "images"
    print(f"   Deneniyor: {val_images_folder}")
    if val_images_folder.exists():
        print("   🔍 Taranıyor...")
        val_normal, val_anomali, val_toplam = klasoru_tara(val_images_folder)
        
        if val_toplam > 0:
            val_normal_ratio = val_normal / val_toplam
            print(f"\n📊 Validation Seti:")
            print(f"   Toplam: {val_toplam} resim")
            print(f"   Sağlıklı (normal): {val_normal} (%{val_normal_ratio*100:.1f})")
            print(f"   Anomalili: {val_anomali} (%{(1-val_normal_ratio)*100:.1f})")
        else:
            print("   ⚠️ Hiç resim bulunamadı!")
    else:
        print(f"   ❌ Valid klasörü bulunamadı!")

# ==================== 4. UYARI KONTROLÜ ====================
if train_toplam > 0 and val_toplam > 0:
    fark = abs(train_normal_ratio - val_normal_ratio)
    print(f"\n⚠️ Normal sınıf oranları arasındaki fark: %{fark*100:.1f}")
    
    if fark > 0.1:  # %10'dan fazla fark varsa
        print("🚨 UYARI: Eğitim ve validation'daki normal/anomali oranları çok farklı!")
        print("   Bu ezberlemeye (overfitting) neden olabilir.")
        print("\n   Çözüm: Stratified split kullanarak veriyi yeniden böl.")
    else:
        print("✅ Oranlar dengeli, ezberleme sorunu büyük ihtimal başka bir yerden kaynaklanıyor.")

print("\n" + "="*50)

# ==================== 5. MODEL TEST (İSTEĞE BAĞLI) ====================
print("\n📷 Model test etmek ister misin?")
cevap = input("   'anold.jpg' için test yapılsın mı? (e/h): ")

if cevap.lower() == 'e':
    print("\n🔍 Model test ediliyor...")
    from ultralytics import YOLO
    import cv2
    
    model = YOLO('best.pt')
    results = model.predict(source='anold.jpg', conf=0.5, save=True)
    
    for r in results:
        im_array = r.plot()
        cv2.imshow("Anomali Tespit Sonucu", im_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()