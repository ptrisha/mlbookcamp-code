import pickle
from flask import Flask
from flask import request
from flask import jsonify


# name the input pickled files
model_file = 'model.bin'
dv_file = 'dv.bin'

# load the model using pickle
with open(model_file, 'rb')as f_in:
    model = pickle.load(f_in)

# load the DictVectorizer using pickle
with open(dv_file, 'rb')as f_in:
    dv = pickle.load(f_in)

app = Flask('score')

@app.route('/predict', methods=['POST'])
def predict():

    client = request.get_json()

    # prepare the input for prediction
    X = dv.transform([client])

    # do the prediction
    y_pred_prob = round(model.predict_proba(X)[0, 1], 3)

    # prepare output
    result =  {
        "card_probability" : float(y_pred_prob)  
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

