# BTK Akademi Datathon 2026 – Career Success Score Prediction

Bu proje, BTK Akademi tarafından düzenlenen Datathon 2026 yarışması kapsamında geliştirilmiştir.

Amaç, öğrencilerin akademik geçmişleri, teknik becerileri, proje deneyimleri, GitHub aktiviteleri ve en önemlisi mentor geri bildirimlerinden (doğal dil metni) yararlanarak `career_success_score` değerini tahmin etmektir.

## 🚀 Model Performance Summary

Aşağıda, yarışma sürecinde geliştirilen ana modelimizin (Super Stacking) CV (Cross Validation) sonuçları yer almaktadır:

| Model | OOF RMSE | Açıklama |
| :--- | :--- | :--- |
| **🏆 Super Stacking Pipeline (Ridge)** | **8.78** | **En başarılı model.** 4 Farklı Base Modelin + Pseudo Labeling'in Ridge ile birleşimi. |
| CatBoost (Text Features Native) | 8.88 | Sadece CatBoost'un dahili NLP motoru ile eğitilmiş model. |
| CatBoost + HuggingFace Embeddings | 8.94 | `paraphrase-multilingual-MiniLM-L12-v2` embeddingleri ile eğitildi. |
| XGBoost + HuggingFace Embeddings | 8.99 | XGBoost Hist tree metodu ile eğitilen model. |
| LightGBM + HuggingFace Embeddings | 9.05 | Optuna ile hiperparametreleri optimize edilmiş LGBM modeli. |

## 🧠 Gelişmiş Yaklaşımlar (Advanced Features)

Bu projede sadece standart algoritmalar değil, Kaggle yarışmalarında fark yaratan üst düzey teknikler kullanılmıştır:

1. **HuggingFace Sentence Embeddings:** `mentor_feedback_text` alanındaki doğal dil verisini sayısal formata çevirmek için `paraphrase-multilingual-MiniLM-L12-v2` transformer modeli kullanılmış ve özellik olarak `embeddings_cache.pkl` dosyasına aktarılmıştır.
2. **Adversarial Validation:** Eğitim (Train) ve Test setleri arasındaki veri dağılım farklılıklarını (distribution shift) önlemek ve overfitting'den kaçınmak amacıyla Adversarial Classifier uygulanmıştır.
3. **Data Leakage & Magic Feature Hunt:** Veride hedef değişkeni ele veren herhangi bir kaçak (leak) olup olmadığı lineer modeller üzerinden test edilmiştir.
4. **Pseudo-Labeling (2-Pass Training):** Test setindeki emin olunan yüksek güvenli tahminler (pseudo-labels) alınarak, ana model 2. aşamada tekrar eğitilmiş ve test setinin yapısı modele öğretilmiştir.
5. **Multi-Seed K-Fold Cross Validation:** Model sonuçlarının stabil olması için 3 farklı random seed (42, 123, 777) ile modeller eğitilip sonuçların ortalaması (averaging) alınmıştır.

## 📂 Proje Yapısı (Pipeline)

Proje, kod okunabilirliğini artırmak ve bilimsel bir süreç izlemek amacıyla adım adım (Pipeline) numaralandırılmıştır:

*   **`01_exploratory_data_analysis.py`:** Keşifçi veri analizi (EDA), değişken dağılımlarının ve zaman tabanlı özelliklerin incelenmesi.
*   **`02_outlier_detection.py`:** Aykırı değer analizi, `career_success_score` dağılımında sınırların belirlenmesi.
*   **`03_target_leakage_hunt.py`:** Hedef sızıntısı testleri, ağırlıklı ortalama deterministik formül avı.
*   **`04_magic_feature_hunt.py`:** Olası gizli alt değişken özelliklerinin lineer ve non-lineer yöntemlerle araştırılması.
*   **`05_adversarial_validation.py`:** Train-Test dağılım kontrolü ve aykırılık tespiti.
*   **`06_optuna_hyperparameter_tuning.py`:** CatBoost ve LightGBM modelleri için geniş uzaylı (night-mode) hiperparametre arayışı.
*   **`07_super_stacking_pipeline.py`:** 4 farklı temel modelin (CatBoost, LightGBM, XGBoost) Ridge regressor ile Stacklendiği, Pseudo-Labeling uygulanan ve final tahmini oluşturan "Super Stacking" ana betiği.

## 🛠️ Kullanılan Teknolojiler

*   **Diller:** Python (Pandas, NumPy)
*   **Modeller:** CatBoost, LightGBM, XGBoost, Scikit-Learn
*   **NLP:** HuggingFace `sentence-transformers`
*   **Optimizasyon:** Optuna (Hyperparameter Tuning)

## 💡 Sonuç
Bu repo, veri ön işlemeden derin NLP vektörizasyonlarına ve uçtan uca Stacking mimarilerine kadar "Winner" tarzı Kaggle çözüm yapısını temsil etmektedir.
