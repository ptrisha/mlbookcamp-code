import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class StudentCourseProfile(BaseModel):
    code_module: str
    code_presentation: str
    gender: str
    region: str
    imd_band: str
    age_band: str
    num_of_prev_attempts: int
    studied_credits: int
    disability: str
    sum_click: float
    mean: float
    max: float
    min: float


model_ref = bentoml.xgboost.get("student_pass_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()
svc = bentoml.Service("student_pass_classifier", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=StudentCourseProfile), output=JSON())
def classify(studentcourse_data):
    application_data = studentcourse_data.dict()
    print(application_data)
    vector = dv.transform(application_data)
    print(vector)
    print(f"Vector.shape: {vector.shape}, vector type: {type(vector)}")
    prediction = model_runner.predict.run(vector)
    print(f"Done prediction")
    result = prediction[0]
    print(f"Prediction result: {result}")
    if result >= 0.5:
        return {"prediction proba": result, "status" : "Pass" }
    else:
        return {"prediction proba": result, "status" : "Does not pass" }

