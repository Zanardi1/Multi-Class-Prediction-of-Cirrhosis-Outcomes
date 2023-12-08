import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC

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
X_train, X_test, y_train, y_test = train_test_split(X, y)

svc = SVC(C=15, probability=True)
cv = cross_val_score(estimator=svc, cv=5, X=X, y=y, scoring='neg_log_loss')

print(sum(cv) / len(cv))
svc.fit(X=X, y=y)
y_pred = svc.predict_proba(test_data)
y_pred = pd.DataFrame(data=y_pred, columns=sample.columns, index=test_data.index)
y_pred.to_csv('submission.csv')
print(y_pred)
