import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')

# Let's find people with very low scores but good features
print("--- Low Scores, Good Features ---")
low_score = train[train['career_success_score'] < 10]
print(f"Total people with score < 10: {len(low_score)}")
if len(low_score) > 0:
    print(low_score[['career_success_score', 'cgpa', 'coding_score', 'communication_score']].head())

# Or high scores but bad features
print("\n--- High Scores, Bad Features ---")
high_score = train[(train['career_success_score'] > 90) & (train['cgpa'] < 2.0)]
print(f"Total people with score > 90 and cgpa < 2.0: {len(high_score)}")

# Let's find standard outliers in target
print("\n--- Outlier thresholds ---")
q1 = train['career_success_score'].quantile(0.25)
q3 = train['career_success_score'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"Lower bound for target: {lower_bound}")

outliers = train[(train['career_success_score'] < lower_bound) | (train['career_success_score'] > upper_bound)]
print(f"Total target outliers: {len(outliers)}")
