#!/usr/bin/env python
# coding: utf-8

import pickle

# name the input pickled files
model_file = 'model1.bin'
dv_file = 'dv.bin'

# load the model using pickle
with open(model_file, 'rb')as f_in:
    model = pickle.load(f_in)

# load the DictVectorizer using pickle
with open(dv_file, 'rb')as f_in:
    dv = pickle.load(f_in)

# client predictor
client = {
    "reports" : 0,
    "share" : 0.001694,
    "expenditure" : 0.12,
    "owner" : "yes"
}

# prepare the input for prediction
X = dv.transform([client])

# do the prediction
y_pred_prob = round(model.predict_proba(X)[0, 1], 3)

# print out the result
print(f"Probability the client gets a credit card: {y_pred_prob}")




