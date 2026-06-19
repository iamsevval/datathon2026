import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Load
train = pd.read_csv('train.csv')
train = train[train['career_success_score'] >= 10.0]
y = train['career_success_score'].copy()
X = train.drop(['student_id', 'career_success_score'], axis=1)

# Basic FE
X['feedback_length'] = X['mentor_feedback_text'].fillna('').apply(len)
X.drop('mentor_feedback_text', axis=1, inplace=True)

# Important interactions
X['project_x_coding'] = X['project_quality_score'] * X['coding_score']
X['project_x_tech_interview'] = X['project_quality_score'] * X['technical_interview_score']

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cat_cols:
    X[c] = X[c].fillna('missing').astype(str)

kf = KFold(n_splits=3, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'iterations': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
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

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15, timeout=300)
print("Best params:", study.best_params)
print("Best MSE:", study.best_value)
