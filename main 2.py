# https://www.kaggle.com/competitions/playground-series-s3e26/discussion/459860

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder


def feature_engineering(df):
    numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'id']
    bins = [0, 30, 60, 100]
    labels = ['Young', 'Middle-aged', 'Elderly']
    df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df['Bilirubin_normal'] = ((df['Bilirubin'] < 1.2) & (df['Age'] >= 18)) | ((df['Bilirubin'] < 1) & (df['Age'] < 18))
    df['Cholesterol_Level'] = pd.cut(df['Cholesterol'], bins=[-float('inf'), 200, 239, float('inf')],
                                     labels=['0', '1', '2'])
    df['Albumin_level'] = np.where(df['Albumin'] < 3.4, 1, np.where(df['Albumin'] > 5.4, 2, 0))
    df['Copper_risk'] = np.where(df['Copper'] > 140, 1, 0)
    df['Alk_Phos_Normal'] = np.where((df['Alk_Phos'] >= 44) & (df['Alk_Phos'] <= 147), 0, 1)
    df['SGOT_Normal'] = np.where((df['SGOT'] >= 8) & (df['SGOT'] <= 45), 0, 1)
    df['Triglycerides_Level'] = pd.cut(df['Tryglicerides'], bins=[-float('inf'), 150, 199, 499, float('inf')],
                                       labels=['0', '1', '2', '3'])
    df['Platelets_Normal'] = np.where((df['Platelets'] > 150000) & (df['Platelets'] <= 450000), 0, 1)
    df['Prothrombin_Normal'] = np.where((df['Prothrombin'] >= 11) & (df['Prothrombin'] <= 13.5), 0, 1)
    for col in numerical_columns:
        df[f'{col}_squared'] = df[col] ** 2
    return df


lgb_params = {'max_depth': 15, 'min_child_samples': 13, 'learning_rate': 0.05285597081335651, 'n_estimators': 294,
              'min_child_weight': 5, 'colsample_bytree': 0.10012816493265511, 'reg_alpha': 0.8767668608061822,
              'reg_lambda': 0.8705834466355764}

raw_data = pd.read_csv('train.csv', index_col=0)
test_data = pd.read_csv('test.csv', index_col=0)
original = pd.read_csv('cirrhosis.csv', index_col=0)
sample = pd.read_csv('sample_submission.csv', index_col=0)
raw_data = pd.concat([original, raw_data])
print(raw_data.head().to_string() + '\n')
print(raw_data.describe().to_string() + '\n')
print(raw_data.info())

numerical_columns = [col for col in raw_data.columns if raw_data[col].dtype in ['int64', 'float64']]
categorical_columns = [col for col in raw_data.columns if raw_data[col].dtype == 'object' and col != 'Status']
test_data_columns = test_data.columns

numerical_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', RobustScaler())])
categorical_preprocessor = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('numerical', numerical_preprocessor, numerical_columns),
                                               ('categorical', categorical_preprocessor, categorical_columns)])

le = LabelEncoder()

X = raw_data.drop(columns=['Status'])
y = raw_data['Status']

X = feature_engineering(X)
test_data = feature_engineering(test_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.fit_transform(X_test)
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
test_data = preprocessor.fit_transform(test_data)

lgb = lgbm.LGBMClassifier(**lgb_params)
early_stopping_callback = lgbm.early_stopping(100, first_metric_only=True, verbose=False)

scores = cross_val_score(estimator=lgb, X=X_train, y=y_train, cv=10, scoring='neg_log_loss')
print(sum(scores) / len(scores))

lgb.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], eval_metric='multi_logloss')
y_pred = lgb.predict_proba(test_data)
y_pred = pd.DataFrame(data=y_pred, columns=sample.columns, index=sample.index)
y_pred.to_csv('Submission.csv')
print(y_pred)

# -0.4475992635933185
