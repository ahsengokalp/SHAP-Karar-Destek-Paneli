🧠 SHAP Tabanlı Karar Destek Sistemi
Bu proje, SHAP (SHapley Additive exPlanations) yöntemiyle çalışan makine öğrenmesi modellerinin kararlarını açıklayabilen, görselleştirebilen ve kullanıcıya içgörü sunan bir karar destek sistemi prototipidir. Örnek senaryo olarak tavuk çiftliklerinde ölüm riski tahmini ele alınmıştır.

🎯 Amaç
Modelin yalnızca sonuç vermesi değil, bu sonucu neden verdiğini kullanıcıya açıklaması hedeflenmiştir. Kullanıcılar tahmin girdilerini ister CSV dosyasıyla ister manuel olarak sağlayabilir. SHAP değerleri sayesinde kararın arkasındaki en etkili değişkenler tablo halinde sunulur ve yorumlanır.

🔧 Özellikler
📥 Kullanıcıdan veri yükleme (CSV veya manuel)

🧠 Farklı modellerle (RandomForest, LightGBM, XGBoost, LogisticRegression) tahmin

📊 SHAP değerleri ile karar açıklaması

🗂 Tahmin ve SHAP geçmişinin Supabase'e kaydı

⬇️ Sonuçları CSV olarak indirme

📡 Qlik Sense'e Webhook üzerinden veri gönderimi

🐔 Örnek Senaryo: Tavuk Ölüm Riski
Sisteme tavuk sayısı, yaş, sıcaklık, aşı durumu gibi bilgiler girildiğinde model, ölüm riskini tahmin eder. Ardından SHAP değerleriyle ölüm kararına en çok etki eden faktörler kullanıcıya görsel ve metinsel olarak açıklanır.

🚀 Kurulum
bash
Kopyala
Düzenle
git clone https://github.com/kullanici_adi/shap-karar-destek.git
cd shap-karar-destek
pip install -r requirements.txt
streamlit run app.py
Not: Supabase bağlantısı için .env dosyasını doğru şekilde yapılandırmayı unutmayın.

📌 Not
Bu sistem, iş zekâsı departmanları için veri odaklı kararlar alınmasını destekleyecek şekilde tasarlanmıştır. Şirket içi analiz süreçlerine şeffaflık kazandırmak ve model kararlarını yorumlanabilir hale getirmek önceliklidir.
