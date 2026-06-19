"""
03. TARGET LEAKAGE HUNT
-----------------------
Bu betik, veri setinde hedefe (career_success_score)
doğrudan işaret eden herhangi bir "veri sızıntısı" 
(data leakage) veya deterministik formül olup olmadığını araştırır.
"""
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

print("--- LEAK HUNTING & FORMULA SEARCH ---")

# 1. Formula Search (Is target just a weighted sum?)
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('career_success_score')
num_cols.remove('student_id') if 'student_id' in num_cols else None

train_clean = train.dropna(subset=num_cols + ['career_success_score'])
X_lin = train_clean[num_cols]
y_lin = train_clean['career_success_score']

lr = LinearRegression()
lr.fit(X_lin, y_lin)
preds = lr.predict(X_lin)
r2 = r2_score(y_lin, preds)

print(f"Linear Regression R^2 (All numeric features): {r2:.4f}")
if r2 > 0.95:
    print("WARNING: TARGET IS ALMOST CERTAINLY A DETERMINISTIC FORMULA!")
    coefs = pd.Series(lr.coef_, index=num_cols).sort_values(ascending=False)
    print("Top Coefficients:")
    print(coefs.head(10))

# 2. Text Search (Does the text contain the target score?)
print("\n--- Text Leak Search ---")
def extract_numbers(text):
    if pd.isna(text): return []
    return [float(x) for x in re.findall(r'\b\d+(?:\.\d+)?\b', str(text))]

train['numbers_in_text'] = train['mentor_feedback_text'].apply(extract_numbers)
# Check if any number exactly matches the target score
matches = []
for idx, row in train.iterrows():
    if row['career_success_score'] in row['numbers_in_text']:
        matches.append(idx)
print(f"Rows where target score explicitly appears in text: {len(matches)} out of {len(train)}")

# 3. Analyze specific groups
print("\n--- Target Variance by University/Role ---")
print(train.groupby('target_role')['career_success_score'].mean())
