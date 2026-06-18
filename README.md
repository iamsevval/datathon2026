# Datathon 2026 - Career Success Score Prediction

Bu depo, BTK Akademi tarafından Kaggle üzerinde düzenlenen **Datathon 2026** yarışmasındaki çözümümü ve analiz süreçlerimi içermektedir. Bu süreç benim katıldığım **ilk Datathon** deneyimiydi ve uçtan uca bir makine öğrenmesi projesi geliştirmek adına inanılmaz öğretici bir yolculuk oldu.

## 🚀 Proje Özeti
Yarışmadaki temel hedef; öğrencilerin akademik, teknik, proje ve sosyal becerilerine ait sayısal verilerin yanı sıra, mentor değerlendirme metinlerini de kullanarak `career_success_score` değerini tahmin etmekti. 

## 🧠 Yaklaşımım ve Öğrendiklerim

Bu ilk datathon deneyimimde standart modellemelerin dışına çıkarak aşağıdaki yöntemleri denedim ve entegre ettim:

- **NLP ile Özellik Çıkarımı (Feature Engineering):** Sadece sayısal verilere bağlı kalmadım. Mentor değerlendirmelerinden oluşan metin verilerinden NLP yöntemleri ile anlamlı özellikler çıkararak modele dahil ettim (`embeddings_cache.pkl`).
- **Gelişmiş Doğrulama (Advanced Validation):** Modelin eğitim verisini ezberlemesini (overfitting) önlemek ve test verisindeki dağılıma ne kadar uyum sağladığını görmek adına `scratch_adv_val.py` ile gelişmiş validation stratejileri kurguladım.
- **Hiperparametre Optimizasyonu:** `optuna_tuner.py` kullanarak modellerimin sınırlarını zorladım ve performansı en üst seviyeye taşımak için hiperparametre aramaları (tuning) yaptım.
- **Derinlemesine Analiz ve Aykırı Değer (Outlier) Tespiti:** Veriyi daha iyi anlamak ve aykırı değerleri yönetmek için spesifik scriptler yazdım (`scratch_analyze.py`, `scratch_find_outliers.py`).
- **Veri Sızıntısı (Data Leak) Araştırması:** Modeli yanıltabilecek potansiyel sızıntıları tespit etmek için araştırmalar yaptım (`scratch_leak_hunt.py`).
- **Sürekli İterasyon:** "Hiçbir ilk deneme mükemmel değildir" prensibiyle yola çıkarak 10'dan fazla versiyon üzerinden (`scratch_explore_v10.py` vb.) algoritmalarımı güncelledim.

## 📂 Proje Yapısı

Repoda bulunan başlıca dosyaların görevleri şu şekildedir:

*   `train_and_predict.py`: Modelin eğitildiği ve nihai tahminlerin (`sample_submission.csv` formatında) üretildiği ana pipeline betiği.
*   `optuna_tuner.py` & `scratch_optuna_search.py`: Optuna ile modellerin hiperparametrelerinin optimize edildiği betikler.
*   `scratch_adv_val.py`: Advanced validation testlerinin yapıldığı dosya.
*   `scratch_find_outliers.py`, `scratch_leak_hunt.py`, `scratch_magic_hunt.py`: Veri seti üzerinde özel analizlerin yapıldığı keşif betikleri.
*   `scratch_analyze.py`, `scratch_explore_v10.py`: Keşifçi veri analizi (EDA) ve versiyonlanmış çalışmaların bulunduğu dosyalar.

## 🛠️ Kullanılan Teknolojiler
*   Python
*   Scikit-Learn, LightGBM, CatBoost, XGBoost (Ensemble Modeller)
*   Optuna (Hiperparametre Optimizasyonu)
*   Pandas, NumPy
*   Doğal Dil İşleme (NLP) Teknikleri

## 💡 Sonuç ve Gelecek Çalışmalar
Bu proje kapsamında kurulan pipeline, tablusal veriler ile serbest metin verilerinin (NLP ile) birleştirilerek anlamlı özellikler üretilmesi noktasında başarılı bir temel oluşturmuştur. Gelişmiş validasyon stratejileri sayesinde modelin genellenebilirliği artırılmıştır. 
Gelecek iterasyonlar için planlanan potansiyel geliştirmeler şunlardır:
- Farklı dil modelleri (örn. BERT, RoBERTa varyantları) ile daha zengin metin özellikleri çıkarılması.
- Model stacking ve blending mimarilerinin daha da derinleştirilmesi.
- Hiperparametre optimizasyon uzayının (search space) genişletilmesi.