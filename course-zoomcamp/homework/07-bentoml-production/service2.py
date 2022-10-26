import numpy as np

import bentoml
from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")
#dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_homework_predictor2", runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(vector):
    #vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)
    
    return prediction
