# Predicting Student Success with OULAD

## Project Objectives
In this project, we train a machine learning model to predict student pass/failure for a course.  This would enable educators to identify potential at-risk students and help in customizing courses to improve student learning experience and outcomes.

## Background
The Open University (OU) is one of the largest distance learning universities in the world.  It is based in the United Kingdom.  The enrollment at the time of publication of the dataset was around 170,000.  Since the OU's launch in 1969, more than 2.3 million people across the world have benefitted from its educational programs.  

Teaching materials and other content are delivered to students online via the university's Virtual Learning Environment (VLE).  Students' interactions with the materials are logged and stored in the OU's data warehouse.  These interactions are realized as clicks on the various resources.

At the OU, courses are known as modules.  There may be  multiple presentations of a given module during the year.  To distinguish the presentations of the same module, each presentation is named by the year and month when it starts.  That is, a presentation that begins in January is labelled 'A', in February labelled 'B', and so on.

The typical duration of a module is 9 months and includes several assessments which are graded by the assigned tutors.  At the end of the module, there is usually a final exam.
  

## Dataset
The dataset which we are using is the Open University Learning Analytics dataset (OULAD), available for download [here](https://analyse.kmi.open.ac.uk/open_dataset/download).

OULAD contains tabular student data from years 2013 to 2014.  Student data includes demographic and module registration information, as well as their assessment results and interactions with the VLE for each student-module-presentation triplet.  Students' VLE interactions are recorded in the form of a summary of their daily activities.

OULAD contains 22 module presentations which consist of B and/or J presentations of 7 modules for years 2013 and 2014.  32,593 students' data are recorded in the dataset.

The download contains the following files:
  
1. assessments.csv  
2. courses.csv
3. studentAssessment.csv
4. studentInfo.csv
5. studentRegistration.csv
6. studentVle.csv
7. vle.csv

## Data and Feature Extraction: Predictor and Target Variables

We are mainly interested in that 3 types of data to use as predictors or independent variables of module outcome: 

1. student demographics - age, gender, region of residence in UK, previous education, etc
2. student scores in the module assessments and exams
3. student activity (number of clicks) on VLE

The target/dependent variable is the module outcome i.e. whether the student passes or does not pass (includes withdrawal) the module.

We aggregate student assessment scores for a module-presentation using the min, max and mean scores.  For student activity on VLE, we sum the total number of clicks logged for the particular module-presentation.

The features/independent variables for analysis are:

  - code\_module: code label for a module  
  - code\_presentation: year and presentation label
  - gender: "M" or "F"
  - region: UK region of residence e.g."South East Region",
  - imd\_band: index of multiple deprivation (a measure of poverty); the higher the band, the better the living conditions e.g. "60-70%",
  - age\_band: age of student e.g. "0-35",
  - num\_of\_prev\_attempts: number of previous attempts at the same module,
  - studied\_credits: total number of credits undertaken,
  - disability: "N" or "Y"
  - sum\_click: total number of clicks on VLE
  - mean: average assessment score,
  - max: maximum assessment score,
  - min: minimum assessment score

The values for the categorical variables can be found in the file named "categorical_vars.txt" in the root folder.

## EDA

The EDA process is found in the Jupyter notebook named "(Part III) EDA" in the "notebooks" folder.

The dataset contains a significant proportion of 'withdrawn' and 'fail' outcomes, so that there is no great imbalance in the dependent variable.

The EDA suggests a relationship between student VLE activity and final outcome.  Other related variables are mean assessment score and imd_band.  The variables that do not appear to have a relationship with final\_result are age, gender, and disability.

## Machine Learning Models

We explored models: LogisticRegression, DecisionTreeClassifier, RandomForest and XGBoostClassifier.  

The model we selected was the XGBoost model.  The exploration and selection of models are recorded in the Jupyter notebook named "(Part IV) Models" in the "notebooks" folder.

The model is saved and deployed to a Docker container using BentoML.  The code for training and saving is found in file named "train.py".

## Local Deployment to Docker Container

Using BentoML, the model is deployed to a Docker container.  The compressed image file (430MB) can be downloaded from [here](https://drive.google.com/file/d/1edbA3JALoFLHAe1QDcsYhjoxbWvQwooX/view?usp=sharing).

Here are the commands for running the downloaded image:  

$ docker load < student\_classifier\_docker.tar.gz  
$ docker run -it --rm -p 3000:3000 student\_pass\_classifier:tleednubz62k7def

Access the Swagger UI on the browser at http://0.0.0.0:3000 .

## Deployment to AWS-Lambda

The service can be accessed at this [endpoint](https://phds14b69i.execute-api.ap-southeast-1.amazonaws.com/). 

The deployment to cloud is done with bentoctl (version 0.3.4).  The deployment configuration files can be found in the folder named "deployment_aws".


## Sample Input/Output for Verifying Model Prediction

These are sample input/output for verifying model prediction on Swagger UI when running the docker image locally or accessing the AWS endpoint.

The ranges of values for categorical variables can be found in the file [categorical\_vars.txt](./categorical_vars.txt) in the root folder.

### Sample 1
{
  "code\_module": "CCC",  
  "code\_presentation": "2014J",
  "gender": "M",
  "region": "South East Region",
  "imd\_band": "60-70%",
  "age\_band": "0-35",
  "num\_of\_prev\_attempts": 0,
  "studied\_credits": 60,
  "disability": "N",
  "sum\_click": 641.0,
  "mean": 67.5625,
  "max": 97.0,
  "min": 40.0
}

Expected result:
{
  "prediction proba": 0.7211555242538452,
  "status": "Pass"
}

## References

Kuzilek, J. _et al._  Open University Learning Analytics dataset.  _Sci. Data_ 4:170171 doi: 10.1038/sdata.2017.171 (2017).









