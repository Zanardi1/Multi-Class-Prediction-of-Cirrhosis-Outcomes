# https://www.kaggle.com/competitions/playground-series-s3e26/discussion/459860

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder


def feature_engineering(df):
    threshold_platelets = 150
    df['thrombocytopenia'] = np.where(df['Platelets'] < threshold_platelets, 1, 0)

    threshold_alk_phos_upper = 147
    threshold_alk_phos_lower = 44
    df['elevated_alk_phos'] = np.where(
        (df['Alk_Phos'] > threshold_alk_phos_upper) | (df['Alk_Phos'] < threshold_alk_phos_lower), 1, 0)

    normal_copper_range = (62, 140)
    df['normal_copper'] = np.where((df['Copper'] >= normal_copper_range[0]) & (df['Copper'] <= normal_copper_range[1]),
                                   1, 0)

    normal_albumin_range = (3.4, 5.4)
    df['normal_albumin'] = np.where(
        (df['Albumin'] >= normal_albumin_range[0]) & (df['Albumin'] < normal_albumin_range[1]), 1, 0)

    normal_bilirubin_range = (0.2, 1.2)
    df['normal_bilirubin'] = np.where(
        (df['Bilirubin'] >= normal_bilirubin_range[0]) & (df['Bilirubin'] <= normal_bilirubin_range[1]), 1, 0)


raw_data = pd.read_csv('train.csv', index_col=0)
test_data = pd.read_csv('test.csv', index_col=0)
sample = pd.read_csv('sample_submission.csv', index_col=0)
print(raw_data.head().to_string() + '\n')
print(raw_data.describe().to_string() + '\n')
print(raw_data.info())

categorical_columns = raw_data.select_dtypes(include=['object', 'bool']).columns
oe = OrdinalEncoder()
raw_data[categorical_columns] = oe.fit_transform(raw_data[categorical_columns])
categorical_columns = categorical_columns.drop(labels=['Status'])
test_data[categorical_columns] = oe.fit_transform(test_data[categorical_columns])

feature_engineering(raw_data)
feature_engineering(test_data)

print(raw_data.head().to_string())
print(raw_data.describe().to_string())
print(raw_data.info())

X = raw_data.drop(columns=['Status'])
y = raw_data['Status']

gbc = GradientBoostingClassifier()
cv = cross_val_score(estimator=gbc, X=X, y=y, cv=20, scoring='neg_log_loss')
print(sum(cv) / len(cv))

gbc.fit(X=X, y=y)
y_pred = gbc.predict_proba(test_data)
y_pred = pd.DataFrame(data=y_pred, columns=sample.columns, index=test_data.index)
y_pred.to_csv('Submission.csv')
print(y_pred)

# -0.44886500419858644
