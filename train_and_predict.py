import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings, time, pickle, os, gc

warnings.filterwarnings('ignore')

print("=" * 70)
print("  V18c: V16b PERFECTED (Multi-Seed + Memory Optimized)")
print("=" * 70)

# ================================================================
# 1. VERİ YÜKLEME
# ================================================================
print("\n[1/7] Veri yükleniyor...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_x.csv')
train_df = train_df[train_df['career_success_score'] >= 10.0]

y = train_df['career_success_score'].copy()
test_ids = test_df['student_id'].copy()

train_df.drop(['student_id', 'career_success_score'], axis=1, inplace=True)
test_df.drop(['student_id'], axis=1, inplace=True)
text_col = 'mentor_feedback_text'

# ================================================================
# 2. FEATURE ENGINEERING
# ================================================================
print("[2/7] Feature Engineering...")
def engineer_features(df):
    df_out = df.copy()
    df_out['feedback_length'] = df_out[text_col].fillna('').apply(len)
    df_out['feedback_word_count'] = df_out[text_col].fillna('').apply(lambda x: len(x.split()))
    tech_cols = ['coding_score', 'problem_solving_score', 'data_structures_score',
                 'sql_score', 'machine_learning_score', 'backend_score', 'frontend_score',
                 'cloud_score', 'devops_score']
    exist_tech = [c for c in tech_cols if c in df_out.columns]
    if exist_tech:
        df_out['total_technical_score'] = df_out[exist_tech].mean(axis=1)
        df_out['max_technical_score'] = df_out[exist_tech].max(axis=1)
        df_out['min_technical_score'] = df_out[exist_tech].min(axis=1)
        df_out['std_technical_score'] = df_out[exist_tech].std(axis=1)
    soft_cols = ['communication_score', 'teamwork_score', 'leadership_score', 'presentation_score']
    exist_soft = [c for c in soft_cols if c in df_out.columns]
    if exist_soft:
        df_out['total_soft_skills_score'] = df_out[exist_soft].mean(axis=1)
    if 'internship_count' in df_out.columns and 'internship_duration_months' in df_out.columns:
        df_out['total_internship_experience'] = df_out['internship_count'] * df_out['internship_duration_months']
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

# ================================================================
# 3. HUGGINGFACE EMBEDDINGS
# ================================================================
print("[3/7] HuggingFace Embeddings yükleniyor...")
with open('embeddings_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
train_emb_np = cache['train']
test_emb_np = cache['test']

emb_cols = [f'emb_{i}' for i in range(train_emb_np.shape[1])]
train_emb = pd.DataFrame(train_emb_np, columns=emb_cols, index=train_fe.index)
test_emb = pd.DataFrame(test_emb_np, columns=emb_cols, index=test_fe.index)

del cache, train_emb_np, test_emb_np
gc.collect()

# ================================================================
# 4. VERİ SETLERİ
# ================================================================
print("[4/7] Veri setleri hazırlanıyor...")
X_A_train = train_fe.copy()
X_A_test = test_fe.copy()
cat_cols_A = X_A_train.select_dtypes(include=['object']).columns.tolist()
cat_cols_A.remove(text_col)
for c in cat_cols_A:
    X_A_train[c] = X_A_train[c].fillna('missing').astype(str)
    X_A_test[c] = X_A_test[c].fillna('missing').astype(str)
X_A_train[text_col] = X_A_train[text_col].fillna('missing').astype(str)
X_A_test[text_col] = X_A_test[text_col].fillna('missing').astype(str)

X_HF_train = pd.concat([train_fe.drop(text_col, axis=1), train_emb], axis=1)
X_HF_test = pd.concat([test_fe.drop(text_col, axis=1), test_emb], axis=1)
cat_cols_HF = X_HF_train.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols_HF:
    X_HF_train[c] = X_HF_train[c].fillna('missing').astype(str).astype('category')
    X_HF_test[c] = X_HF_test[c].fillna('missing').astype(str).astype('category')

del train_fe, test_fe, train_emb, test_emb
gc.collect()

# ================================================================
# 5. MULTI-SEED PIPELINE
# ================================================================
SEEDS = [42, 123, 777]
all_seed_preds = []

for si, SEED in enumerate(SEEDS):
    print(f"\n{'='*50}")
    print(f"  SEED {si+1}/{len(SEEDS)}: {SEED}")
    print(f"{'='*50}")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_A = np.zeros(len(y)); test_A = np.zeros(len(X_A_test))
    oof_B = np.zeros(len(y)); test_B = np.zeros(len(X_HF_test))
    oof_C = np.zeros(len(y)); test_C = np.zeros(len(X_HF_test))
    oof_D = np.zeros(len(y)); test_D = np.zeros(len(X_HF_test))

    # A: CatBoost Native Text
    t0 = time.time()
    for fold, (trn, val) in enumerate(kf.split(X_A_train, y)):
        m = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3,
                              cat_features=cat_cols_A, text_features=[text_col],
                              random_state=SEED, verbose=0, early_stopping_rounds=100)
        m.fit(X_A_train.iloc[trn], y.iloc[trn], eval_set=(X_A_train.iloc[val], y.iloc[val]))
        oof_A[val] = m.predict(X_A_train.iloc[val])
        test_A += m.predict(X_A_test) / 5
        del m; gc.collect()
    print(f"  -> Model A: CV {mean_squared_error(y, oof_A):.4f} ({time.time()-t0:.0f}s)")

    # B: CatBoost + HF
    t0 = time.time()
    for fold, (trn, val) in enumerate(kf.split(X_HF_train, y)):
        m = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                              cat_features=cat_cols_HF, random_state=SEED, verbose=0)
        m.fit(X_HF_train.iloc[trn], y.iloc[trn], eval_set=(X_HF_train.iloc[val], y.iloc[val]), early_stopping_rounds=100)
        oof_B[val] = m.predict(X_HF_train.iloc[val])
        test_B += m.predict(X_HF_test) / 5
        del m; gc.collect()
    print(f"  -> Model B: CV {mean_squared_error(y, oof_B):.4f} ({time.time()-t0:.0f}s)")

    # C: LightGBM + HF
    t0 = time.time()
    for fold, (trn, val) in enumerate(kf.split(X_HF_train, y)):
        m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, max_depth=8,
                              num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X_HF_train.iloc[trn], y.iloc[trn])
        oof_C[val] = m.predict(X_HF_train.iloc[val])
        test_C += m.predict(X_HF_test) / 5
        del m; gc.collect()
    print(f"  -> Model C: CV {mean_squared_error(y, oof_C):.4f} ({time.time()-t0:.0f}s)")

    # D: XGBoost + HF
    t0 = time.time()
    X_HF_train_xgb = X_HF_train.copy()
    X_HF_test_xgb = X_HF_test.copy()
    for c in cat_cols_HF:
        X_HF_train_xgb[c] = X_HF_train_xgb[c].astype('category')
        X_HF_test_xgb[c] = X_HF_test_xgb[c].astype('category')
    for fold, (trn, val) in enumerate(kf.split(X_HF_train_xgb, y)):
        m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=-1, enable_categorical=True, tree_method='hist')
        m.fit(X_HF_train_xgb.iloc[trn], y.iloc[trn], verbose=0)
        oof_D[val] = m.predict(X_HF_train_xgb.iloc[val])
        test_D += m.predict(X_HF_test_xgb) / 5
        del m; gc.collect()
    del X_HF_train_xgb, X_HF_test_xgb; gc.collect()
    print(f"  -> Model D: CV {mean_squared_error(y, oof_D):.4f} ({time.time()-t0:.0f}s)")

    # RIDGE STACKING
    meta_train = np.column_stack([oof_A, oof_B, oof_C, oof_D])
    meta_test = np.column_stack([test_A, test_B, test_C, test_D])
    ridge = Ridge(alpha=100.0, random_state=SEED)
    ridge.fit(meta_train, y)
    pass1_test = ridge.predict(meta_test)
    print(f"  -> Stacked CV: {mean_squared_error(y, ridge.predict(meta_train)):.4f}")

    # PSEUDO-LABELING (Pass 2, B, C, D only)
    pseudo_y = np.clip(pass1_test, 0, 100)
    X_HF_combined = pd.concat([X_HF_train, X_HF_test], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y.reset_index(drop=True), pd.Series(pseudo_y)], axis=0).reset_index(drop=True)
    sw = np.concatenate([np.ones(len(y)), np.ones(len(pseudo_y)) * 0.5])

    test_B2 = np.zeros(len(X_HF_test))
    test_C2 = np.zeros(len(X_HF_test))
    test_D2 = np.zeros(len(X_HF_test))
    kf2 = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (trn, val) in enumerate(kf2.split(X_HF_combined, y_combined)):
        m = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                              cat_features=cat_cols_HF, random_state=SEED, verbose=0)
        m.fit(X_HF_combined.iloc[trn], y_combined.iloc[trn], sample_weight=sw[trn],
              eval_set=(X_HF_combined.iloc[val], y_combined.iloc[val]), early_stopping_rounds=100, verbose=0)
        test_B2 += m.predict(X_HF_test) / 5
        del m; gc.collect()

    for fold, (trn, val) in enumerate(kf2.split(X_HF_combined, y_combined)):
        m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, max_depth=8,
                              num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X_HF_combined.iloc[trn], y_combined.iloc[trn], sample_weight=sw[trn])
        test_C2 += m.predict(X_HF_test) / 5
        del m; gc.collect()

    X_HF_comb_xgb = X_HF_combined.copy()
    X_HF_test_xgb2 = X_HF_test.copy()
    for c in cat_cols_HF:
        X_HF_comb_xgb[c] = X_HF_comb_xgb[c].astype('category')
        X_HF_test_xgb2[c] = X_HF_test_xgb2[c].astype('category')
    for fold, (trn, val) in enumerate(kf2.split(X_HF_comb_xgb, y_combined)):
        m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=-1, enable_categorical=True, tree_method='hist')
        m.fit(X_HF_comb_xgb.iloc[trn], y_combined.iloc[trn], sample_weight=sw[trn], verbose=0)
        test_D2 += m.predict(X_HF_test_xgb2) / 5
        del m; gc.collect()
    
    del X_HF_combined, X_HF_comb_xgb, y_combined, sw; gc.collect()

    meta_test_p2 = np.column_stack([test_A, test_B2, test_C2, test_D2]) 
    pass2_test = ridge.predict(meta_test_p2)
    seed_preds = 0.6 * pass2_test + 0.4 * pass1_test
    all_seed_preds.append(seed_preds)
    print(f"  -> Seed {SEED} tamamlandı!")

# ================================================================
# 6. MULTI-SEED AVERAGING
# ================================================================
print(f"\n[6/7] Multi-Seed Averaging...")
final_preds = np.mean(all_seed_preds, axis=0)
final_preds = np.clip(final_preds, 0, 100)

submission = pd.DataFrame({'student_id': test_ids, 'career_success_score': final_preds})
submission.to_csv('sample_submission.csv', index=False)

print("\n" + "=" * 70)
print("  V18c PERFECTED TAMAMLANDI!")
print("  'sample_submission.csv' başarıyla oluşturuldu.")
print("=" * 70)
