# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pickle

# %%
C=1.0
n_splits = 5
output_file = f'model_C={C}.bin'

# %%
url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(url) 

# %%
df.columns = df.columns.str.lower().str.replace(" ", "_")
str_cols = list(df.select_dtypes('str'))
for c in str_cols:
    df[c] = df[c].str.lower().str.replace(" ", "_")
df.churn = (df.churn=='yes').astype(int)
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# %%
numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

def train(df, y, C):
    dict = df[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dict)
    model = LogisticRegression(max_iter=10000, C=C)
    model.fit(X, y)
    return dv, model

def predict(df, dv, model):
    dict = df[categorical+numerical].to_dict(orient='records')
    X = dv.transform(dict)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred


# %%
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
print('validating')
scores = []
fold = 0
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train, C)
    y_pred = predict(df_val, dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    
    print(f'fold {fold}: {auc}')
    fold = fold +1
print(f'C={C}: {round(np.mean(scores),3)} +- {round(np.std(scores),3)}')

# %%
print('training final model')
y_full_train = df_full_train.churn.values
y_test = df_test.churn.values
dv, model = train(df_full_train, y_full_train, C)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)
print(f'auc: {auc}')


# %%
with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)
print(f"Model saved to file {output_file}")


# %%



