# https://www.kaggle.com/competitions/playground-series-s3e26/discussion/459860

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def feature_engineering(df):
    numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'id']
    bins = [0, 30, 60, 100]
    labels = ['Young', 'Middle-aged', 'Elderly']
    df['Age group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df['Bilirubin normal'] = ((df['Bilirubin'] < 1.2) & (df['Age'] >= 18)) | ((df['Bilirubin'] < 1) & (df['Age'] < 18))
    df['Cholesterol Level'] = pd.cut(df['Cholesterol'], bins=[-float('inf'), 200, 239, float('inf')],
                                     labels=['0', '1', '2'])
    df['Albumin level'] = np.where(df['Albumin'] < 3.4, 1, np.where(df['Albumin'] > 5.4, 2, 0))
    df['Copper risk'] = np.where(df['Copper'] > 140, 1, 0)
    df['Alk Phos Normal'] = np.where((df['Alk_Phos'] >= 44) & (df['Alk_Phos'] <= 147), 0, 1)
    df['SGOT Normal'] = np.where((df['SGOT'] >= 8) & (df['SGOT'] <= 45), 0, 1)
    df['Triglycerides Level'] = pd.cut(df['Tryglicerides'], bins=[-float('inf'), 150, 199, 499, float('inf')],
                                       labels=['0', '1', '2', '3'])
    df['Platelets Normal'] = np.where((df['Platelets'] > 150000) & (df['Platelets'] <= 450000), 0, 1)
    df['Prothrombin Normal'] = np.where((df['Prothrombin'] >= 11) & (df['Prothrombin'] <= 13.5), 0, 1)
    for col in numerical_columns:
        df[f'{col}_squared'] = df[col] ** 2
    return df


lgb_params = {'max_depth': 15, 'min_child_samples': 13, 'learning_rate': 0.05285597081335651, 'n_estimators': 294,
              'min_child_weight': 5, 'colsample_bytree': 0.10012816493265511, 'reg_alpha': 0.8767668608061822,
              'reg_lambda': 0.8705834466355764}

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

X = raw_data.drop(columns=['Status'])
y = raw_data['Status']

X = feature_engineering(X)
test_data = feature_engineering(test_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

lgb = lgbm.LGBMClassifier(**lgb_params)
early_stopping_callback = lgbm.early_stopping(100, first_metric_only=True, verbose=False)

scores = cross_val_score(estimator=lgb, X=X_train, y=y_train, cv=10, scoring='neg_log_loss')
print(sum(scores) / len(scores))

lgb.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], eval_metric='multi_logloss')
y_pred = lgb.predict_proba(test_data)
y_pred = pd.DataFrame(data=y_pred, columns=sample.columns, index=test_data.index)
y_pred.to_csv('Submission.csv')
print(y_pred)

# -0.44886500419858644
