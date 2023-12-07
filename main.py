import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('train.csv', index_col=0)
print(raw_data.head().to_string() + '\n')
print(raw_data.describe().to_string() + '\n')
print(raw_data.info())

categorical_columns = raw_data.select_dtypes(include=['object', 'bool']).columns
oe = OrdinalEncoder()
raw_data[categorical_columns] = oe.fit_transform(raw_data[categorical_columns])

X = raw_data.drop(columns=['Status'])
y = raw_data['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y)
pass