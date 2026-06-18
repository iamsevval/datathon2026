import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

train = pd.read_csv('train.csv')
train = train[train['career_success_score'] > 10.0]

print("--- HUNTING THE MAGIC FORMULA ---")

score_cols = [c for c in train.columns if 'score' in c and c != 'career_success_score']
print("Score columns:", score_cols)

# Try linear regression JUST on score columns
X_scores = train[score_cols].fillna(0)
y = train['career_success_score']

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_scores, y)
preds = lr.predict(X_scores)
print(f"\nR^2 using ONLY sub-scores (Linear): {r2_score(y, preds):.4f}")

# Look at coefficients
coefs = pd.Series(lr.coef_, index=score_cols).sort_values(ascending=False)
print("Top linear coefficients:")
print(coefs)

# Try RandomForest to see if there's a non-linear perfect relationship
rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_scores, y)
rf_preds = rf.predict(X_scores)
print(f"\nR^2 using ONLY sub-scores (RandomForest): {r2_score(y, rf_preds):.4f}")

# What if we add numerical features like age, counts?
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('career_success_score')
if 'student_id' in num_cols: num_cols.remove('student_id')

X_num = train[num_cols].fillna(0)
rf.fit(X_num, y)
rf_preds_num = rf.predict(X_num)
print(f"\nR^2 using ALL NUMERIC (RandomForest): {r2_score(y, rf_preds_num):.4f}")

importances = pd.Series(rf.feature_importances_, index=num_cols).sort_values(ascending=False)
print("\nTop 10 Feature Importances in RF:")
print(importances.head(10))

# Try to find target leak in categorical
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
if 'student_id' in cat_cols: cat_cols.remove('student_id')
if 'mentor_feedback_text' in cat_cols: cat_cols.remove('mentor_feedback_text')

for c in cat_cols:
    means = train.groupby(c)['career_success_score'].mean()
    if means.std() > 5:
        print(f"\nHUGE Target Shift in Category {c}:")
        print(means)
