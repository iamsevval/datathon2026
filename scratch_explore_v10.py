import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

print("--- Train Columns ---")
print(train.columns.tolist())

print("\n--- Year Analysis ---")
train['study_duration'] = train['graduation_year'] - train['application_year']
test['study_duration'] = test['graduation_year'] - test['application_year']

train['years_since_grad'] = 2026 - train['graduation_year']
test['years_since_grad'] = 2026 - test['graduation_year']

print("Train study_duration distribution:")
print(train['study_duration'].value_counts(normalize=True).head())
print("Test study_duration distribution:")
print(test['study_duration'].value_counts(normalize=True).head())

print("\nTrain years_since_grad distribution:")
print(train['years_since_grad'].value_counts(normalize=True).head())
print("Test years_since_grad distribution:")
print(test['years_since_grad'].value_counts(normalize=True).head())

