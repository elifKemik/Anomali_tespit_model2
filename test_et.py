from ultralytics import YOLO

# Modelini yükle
model = YOLO('best.pt') 

# Tahmin yap (Düşük güven eşiği ile her şeyi görelim)
results = model.predict(source='cerebellah-hypoplasia-20a_aug_3_png_jpg.rf.4ab11ade67d18fe8b92c8f02ea4e37e2.jpg', conf=0.05)

print("\n--- TAHMİN SONUÇLARI ---")

# Sonuçları terminale yazdır
for r in results:
    for box in r.boxes:
        # Sınıf ismi
        class_id = int(box.cls[0])
        label = model.names[class_id]
        
        # Güven skoru (Confidence)
        conf = float(box.conf[0])
        
        print(f"Hastalık: {label} | Tahmin Değeri: %{conf*100:.2f}")

if len(results[0].boxes) == 0:
    print("Herhangi bir hastalık tespit edilemedi.")

print("------------------------\n")