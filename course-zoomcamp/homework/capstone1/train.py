
# Trains the final XGBoost model and saves the model using BentoML

# Import the libraries
import os

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# read in data and process data
df = pd.read_csv("data/merged.csv", index_col=0)


# Replace the data item '55<=' in 'age_band' column with '55+' as '<' and '=' chars are not accepted in some models
df.replace({'age_band': {'55<=': '55+'}}, inplace=True)

# Replace value '10-20' in 'imd_band' column with '10-20%' for input data consistency
df.replace({'imd_band': {'10-20': '10-20%'}}, inplace=True)

df.imd_band.value_counts()


# Prepare the depedent variable by replacing categorical values with numerical values

di = {'Pass':1, 'Distinction': 1, 'Withdrawn': 0, 'Fail':0}
df['final_result'] = df['final_result'].map(di)
df.final_result.value_counts()


# Prepare the independent variables: we'll drop 'id_student' and highest_education' columns

categorical = ['code_module', 'code_presentation', 'gender', 'region', 'imd_band', 'age_band', 'disability']
numerical = ['num_of_prev_attempts', 'studied_credits', 'sum_click', 'mean', 'max', 'min']

df.drop( ['id_student', 'highest_education'], axis=1, inplace=True)

print(df.columns)

# Split data into train and test sets

df_train, df_test = train_test_split(df, test_size=0.2, random_state=11, stratify=df['final_result'])

print( df_train.shape, df_test.shape )

# separate the independent and dependent data, and drop indexes
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.final_result
y_test = df_test.final_result

del df_train['final_result']
del df_test['final_result']


# one-hot encoding for categorical independent variables; resulting matrix should have 48 columns

dv = DictVectorizer(sparse=False)

train_dict = df_train.fillna(0).to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

test_dict = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(test_dict)

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

print(f"X_train columns: {df_train.columns}")


# Run the selected model
xgb_params = {
    'eta': 0.1, 
    'max_depth': 6,
    'min_child_weight': 30,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 2,
    'seed': 1,
    'verbosity': 1,
}


dtrain = xgb.DMatrix(X_train, label=y_train) # feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test) #, feature_names=features)

model = xgb.train(xgb_params, dtrain, num_boost_round=130)

y_pred = model.predict(dtest)
roc_score = roc_auc_score(y_test, y_pred)

print(f"ROC: {roc_score}")

# obtain example input, expected output, and prediction value for testing the saved model
import json
request = df_test.iloc[45].to_dict()
print("Example JSON input X_train[45]:")
print(json.dumps(request, indent=2))
print(f"y_test[45]: {y_test[45]}")
print(f"Prediction y_pred[45]: {y_pred[45]}")



# save model and dictVectorizer using Bentoml
import bentoml

bentoml.xgboost.save_model("student_pass_model", model, custom_objects={ "dictVectorizer" : dv } )




