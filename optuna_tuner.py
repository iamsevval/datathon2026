import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Kaggle V11 THE OVERNIGHT OPTUNA CHAMPION BAŞLATILIYOR...")

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_x.csv')

# --- OUTLIER TEMİZLİĞİ ---
train_df = train_df[train_df['career_success_score'] >= 10.0]

y = train_df['career_success_score'].copy()
train_ids = train_df['student_id'].copy()
test_ids = test_df['student_id'].copy()

train_df.drop(['student_id', 'career_success_score'], axis=1, inplace=True)
test_df.drop(['student_id'], axis=1, inplace=True)

# --- ZAMAN ÖZELLİKLERİ (NUMERİK OLMAK ZORUNDA!) ---
# Extrapolasyon yapılabilmesi için yıllar int/float kalmalı!
if 'application_year' in train_df.columns and 'graduation_year' in train_df.columns:
    train_df['study_duration'] = train_df['graduation_year'] - train_df['application_year']
    test_df['study_duration'] = test_df['graduation_year'] - test_df['application_year']
    
    train_df['years_since_grad'] = 2026 - train_df['graduation_year']
    test_df['years_since_grad'] = 2026 - test_df['graduation_year']

# --- DEEP NLP ---
print("Deep NLP Vektörleri Hesaplanıyor...")
text_col = 'mentor_feedback_text'
all_text = pd.concat([train_df[text_col], test_df[text_col]]).fillna('').tolist()

st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
all_embeddings = st_model.encode(all_text, show_progress_bar=False)

emb_cols = [f'text_emb_{i}' for i in range(all_embeddings.shape[1])]
train_emb_df = pd.DataFrame(all_embeddings[:len(train_df)], columns=emb_cols, index=train_df.index)
test_emb_df = pd.DataFrame(all_embeddings[len(train_df):], columns=emb_cols, index=test_df.index)

def engineer_features(df):
    df_out = df.copy()
    df_out['feedback_length'] = df_out[text_col].fillna('').apply(len)
    
    # Proje Sinerjisi
    if 'project_quality_score' in df_out.columns:
        if 'coding_score' in df_out.columns:
            df_out['synergy_project_coding'] = df_out['project_quality_score'] * df_out['coding_score']
        if 'technical_interview_score' in df_out.columns:
            df_out['synergy_project_tech_interview'] = df_out['project_quality_score'] * df_out['technical_interview_score']
            
    tech_cols = ['coding_score', 'problem_solving_score', 'data_structures_score', 
                 'sql_score', 'machine_learning_score', 'backend_score', 'frontend_score', 
                 'cloud_score', 'devops_score']
    exist_tech = [c for c in tech_cols if c in df_out.columns]
    if exist_tech:
        df_out['total_technical_score'] = df_out[exist_tech].mean(axis=1)
        
    if 'internship_count' in df_out.columns and 'internship_duration_months' in df_out.columns:
        df_out['total_internship_experience'] = df_out['internship_count'] * df_out['internship_duration_months']
        
    df_out.drop(text_col, axis=1, inplace=True)
    return df_out

train_fe = engineer_features(train_df)
test_fe = engineer_features(test_df)

all_fe = pd.concat([train_fe, test_fe], axis=0)
group_cols = ['university_tier', 'department', 'target_role']
calc_cols = ['coding_score', 'portfolio_score', 'total_technical_score', 'cgpa']

for g in group_cols:
    if g in all_fe.columns:
        for c in calc_cols:
            if c in all_fe.columns:
                mean_val = all_fe.groupby(g)[c].transform('mean')
                all_fe[f'{c}_ratio_to_{g}_mean'] = all_fe[c] / (mean_val + 1e-5)

train_fe = all_fe.iloc[:len(train_df)].copy()
test_fe = all_fe.iloc[len(train_df):].copy()

X = pd.concat([train_fe, train_emb_df], axis=1)
X_test = pd.concat([test_fe, test_emb_df], axis=1)

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cat_cols:
    X[c] = X[c].fillna('missing').astype(str)
    X_test[c] = X_test[c].fillna('missing').astype(str)
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- OPTUNA TUNING (GECE MESAİSİ) BAŞLIYOR ---")
print("1. CatBoost Optimizasyonu...")
def cat_objective(trial):
    params = {
        'iterations': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'cat_features': cat_cols,
        'verbose': 0,
        'random_state': 42
    }
    oof = np.zeros(len(X))
    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0)
        oof[val_idx] = model.predict(X_va)
    return mean_squared_error(y, oof)

cat_study = optuna.create_study(direction='minimize')
cat_study.optimize(cat_objective, n_trials=15)
best_cat_params = cat_study.best_params
best_cat_params['iterations'] = 1500
best_cat_params['cat_features'] = cat_cols
best_cat_params['random_state'] = 42
best_cat_params['verbose'] = 0

print("2. LightGBM Optimizasyonu...")
def lgb_objective(trial):
    params = {
        'n_estimators': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }
    oof = np.zeros(len(X))
    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_va)
    return mean_squared_error(y, oof)

lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=15)
best_lgb_params = lgb_study.best_params
best_lgb_params['n_estimators'] = 1500
best_lgb_params['n_jobs'] = -1
best_lgb_params['verbose'] = -1
best_lgb_params['random_state'] = 42

print("\n--- NİHAİ MODELLERİN EĞİTİMİ (EN İYİ PARAMETRELERLE) ---")
oof_cat = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))
lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

print("   -> Tuned CatBoost Eğitiliyor...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    model_cat = CatBoostRegressor(**best_cat_params)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, verbose=0)
    oof_cat[val_idx] = model_cat.predict(X_va)
    cat_preds += model_cat.predict(X_test) / 5

print("   -> Tuned LightGBM Eğitiliyor...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    model_lgb = lgb.LGBMRegressor(**best_lgb_params)
    model_lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    lgb_preds += model_lgb.predict(X_test) / 5

print("   -> XGBoost Eğitiliyor (Sabit Kaggle Parametreleri ile)...")
xgb_params = {'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 6, 'subsample': 0.8, 
              'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'enable_categorical': True, 'tree_method': 'hist'}
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, verbose=0)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    xgb_preds += model_xgb.predict(X_test) / 5

print("\n--- RIDGE STACKING (OPTUNA TUNED) ---")
meta_X_train = np.column_stack([oof_cat, oof_lgb, oof_xgb])
meta_X_test = np.column_stack([cat_preds, lgb_preds, xgb_preds])

meta_model = Ridge(alpha=10.0, random_state=42)
meta_model.fit(meta_X_train, y)

final_oof_preds = meta_model.predict(meta_X_train)
best_mse = mean_squared_error(y, final_oof_preds)
print(f"\n!!! OVERNIGHT OPTUNA CHAMPION FINAL OOF MSE: {best_mse:.2f} !!!")

final_test_preds = meta_model.predict(meta_X_test)
final_test_preds = np.clip(final_test_preds, 0, 100)

submission = pd.DataFrame({
    'student_id': test_ids,
    'career_success_score': final_test_preds
})

submission.to_csv('sample_submission.csv', index=False)
print("\nYarınki zafer dosyası ('sample_submission.csv') taptaze hazırlandı!")
