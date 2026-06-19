import pandas as pd
import json

train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

print("--- Data Shift Analysis ---")
for col in ['application_year', 'graduation_year', 'department', 'university_tier', 'hobby', 'target_role']:
    if col in train.columns and col in test.columns:
        train_vals = set(train[col].dropna().unique())
        test_vals = set(test[col].dropna().unique())
        print(f"\n{col}:")
        print(f"  Train Categories: {len(train_vals)}")
        print(f"  Test Categories: {len(test_vals)}")
        
        diff_test = test_vals - train_vals
        if diff_test:
            print(f"  Categories in Test NOT in Train: {len(diff_test)}")
            print(f"  Examples: {list(diff_test)[:5]}")
        else:
            print("  No unseen categories in test.")

print("\n--- Target Stats ---")
print(train['career_success_score'].describe())

print("\n--- Correlation with Target (Top 15) ---")
numeric_df = train.select_dtypes(include=['int64', 'float64'])
corr = numeric_df.corr()['career_success_score'].abs().sort_values(ascending=False)
print(corr.head(16))
