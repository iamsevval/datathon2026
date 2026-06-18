import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

# Drop target
train.drop('career_success_score', axis=1, inplace=True)
train.drop('student_id', axis=1, inplace=True)
test.drop('student_id', axis=1, inplace=True)

# Add Target for Adversarial Validation: 0 for train, 1 for test
train['is_test'] = 0
test['is_test'] = 1

df = pd.concat([train, test], axis=0).reset_index(drop=True)
y = df['is_test']
X = df.drop('is_test', axis=1)

# Basic feature engineering to match our train_and_predict
text_col = 'mentor_feedback_text'
X['feedback_length'] = X[text_col].fillna('').apply(len)
X.drop(text_col, axis=1, inplace=True)

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cat_cols:
    X[c] = X[c].fillna('missing').astype(str)

X.drop(['application_year', 'graduation_year'], axis=1, inplace=True, errors='ignore')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=4, 
                               cat_features=cat_cols, verbose=0, random_state=42)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=20, verbose=0)
    oof[val_idx] = model.predict_proba(X_va)[:, 1]

auc = roc_auc_score(y, oof)
print(f"Adversarial Validation AUC: {auc:.4f}")

if auc > 0.55:
    print("\nWarning: Train and Test have different distributions! Features causing this:")
    # Train on full to get feature importances
    model.fit(X, y, verbose=0)
    importances = model.get_feature_importance()
    feat_imps = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
    print(feat_imps.head(10))
else:
    print("Train and Test sets are perfectly mixed (No distribution shift).")
