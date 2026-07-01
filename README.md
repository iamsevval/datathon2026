# BTK Akademi Datathon 2026 – Career Success Score Prediction

This project was developed as part of the Datathon 2026 competition organized by BTK Akademi.

The goal is to predict the `career_success_score` value by leveraging students' academic backgrounds, technical skills, project experience, GitHub activity, and — most importantly — mentor feedback (natural language text).

## 🚀 Model Performance Summary

Below are the CV (Cross Validation) results of our main model (Super Stacking) developed during the competition:

| Model | OOF RMSE | Description |
| :--- | :--- | :--- |
| **🏆 Super Stacking Pipeline (Ridge)** | **8.78** | **Best performing model.** A combination of 4 different Base Models + Pseudo Labeling, stacked with Ridge. |
| CatBoost (Text Features Native) | 8.88 | Model trained using only CatBoost's built-in NLP engine. |
| CatBoost + HuggingFace Embeddings | 8.94 | Trained with `paraphrase-multilingual-MiniLM-L12-v2` embeddings. |
| XGBoost + HuggingFace Embeddings | 8.99 | Model trained with the XGBoost Hist tree method. |
| LightGBM + HuggingFace Embeddings | 9.05 | LGBM model with hyperparameters optimized via Optuna. |

## 🧠 Advanced Approaches (Advanced Features)

This project uses not only standard algorithms, but also advanced techniques that make a difference in Kaggle competitions:

1. **HuggingFace Sentence Embeddings:** The `paraphrase-multilingual-MiniLM-L12-v2` transformer model was used to convert the natural language data in the `mentor_feedback_text` field into numerical format, and exported as a feature into the `embeddings_cache.pkl` file.
2. **Adversarial Validation:** An Adversarial Classifier was applied to prevent distribution shift between the Train and Test sets and to avoid overfitting.
3. **Data Leakage & Magic Feature Hunt:** The data was tested via linear models to check for any leaks that could give away the target variable.
4. **Pseudo-Labeling (2-Pass Training):** High-confidence predictions (pseudo-labels) from the test set were taken, and the main model was retrained in a 2nd stage, teaching the model the structure of the test set.
5. **Multi-Seed K-Fold Cross Validation:** To ensure stable model results, models were trained with 3 different random seeds (42, 123, 777) and the results were averaged.

## 📂 Project Structure (Pipeline)

The project is numbered step by step (Pipeline) to improve code readability and follow a scientific process:

*   **`01_exploratory_data_analysis.py`:** Exploratory data analysis (EDA), examination of variable distributions and time-based features.
*   **`02_outlier_detection.py`:** Outlier analysis, determining the boundaries in the `career_success_score` distribution.
*   **`03_target_leakage_hunt.py`:** Target leakage tests, hunting for a weighted-average deterministic formula.
*   **`04_magic_feature_hunt.py`:** Investigating possible hidden sub-variable features using linear and non-linear methods.
*   **`05_adversarial_validation.py`:** Train-Test distribution check and outlier detection.
*   **`06_optuna_hyperparameter_tuning.py`:** Wide-space (night-mode) hyperparameter search for CatBoost and LightGBM models.
*   **`07_super_stacking_pipeline.py`:** The main "Super Stacking" script, where 4 different base models (CatBoost, LightGBM, XGBoost) are stacked with a Ridge regressor, Pseudo-Labeling is applied, and the final prediction is produced.

## 🛠️ Technologies Used

*   **Languages:** Python (Pandas, NumPy)
*   **Models:** CatBoost, LightGBM, XGBoost, Scikit-Learn
*   **NLP:** HuggingFace `sentence-transformers`
*   **Optimization:** Optuna (Hyperparameter Tuning)
