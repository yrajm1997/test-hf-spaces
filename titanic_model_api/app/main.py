import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import pandas as pd
from titanic_model.processing.data_manager import load_dataset, load_pipeline
from titanic_model import __version__ as _version
from titanic_model.config.core import config
from sklearn.model_selection import train_test_split
from titanic_model.predict import make_prediction

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# FastAPI object
app = FastAPI()


################################# Prometheus related code START ######################################################
import prometheus_client as prom

acc_metric = prom.Gauge('titanic_accuracy_score', 'Accuracy score for few random 100 test samples')
f1_metric = prom.Gauge('titanic_f1_score', 'F1 score for few random 100 test samples')
precision_metric = prom.Gauge('titanic_precision_score', 'Precision score for few random 100 test samples')
recall_metric = prom.Gauge('titanic_recall_score', 'Recall score for few random 100 test samples')

# LOAD TEST DATA
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
titanic_pipe= load_pipeline(file_name=pipeline_file_name)
data = load_dataset(file_name=config.app_config.training_data_file)    # read complete data

X_train, X_test, y_train, y_test = train_test_split(                   # divide into train and test set
    data[config.model_config.features],
    data[config.model_config.target],
    test_size=config.model_config.test_size,
    random_state=config.model_config.random_state,
)
test_data = X_test.copy()
test_data['target'] = y_test.values


# Function for updating metrics
def update_metrics():
    global test_data
    # Performance on test set
    size = random.randint(100, 130)
    test = test_data.sample(size, random_state = random.randint(0, 1e6))       # sample few 100 rows randomly
    y_pred = titanic_pipe.predict(test.iloc[:, :-1])                           # prediction
    acc = accuracy_score(test['target'], y_pred).round(3)                    # accuracy score
    f1 = f1_score(test['target'], y_pred).round(3)                           # F1 score
    precision = precision_score(test['target'], y_pred).round(3)             # Precision score
    recall = recall_score(test['target'], y_pred).round(3)                   # Recall score
    
    acc_metric.set(acc)
    f1_metric.set(f1)
    precision_metric.set(precision)
    recall_metric.set(recall)

@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())

################################# Prometheus related code END ######################################################


# UI - Input components
in_Pid = gradio.Textbox(lines=1, placeholder=None, value="79", label='Passenger Id')
in_Pclass = gradio.Radio(['1', '2', '3'], type="value", label='Passenger class')
in_Pname = gradio.Textbox(lines=1, placeholder=None, value="Caldwell, Master. Alden Gates", label='Passenger Name')
in_sex = gradio.Radio(["Male", "Female"], type="value", label='Gender')
in_age = gradio.Textbox(lines=1, placeholder=None, value="14", label='Age of the passenger in yrs')
in_sibsp = gradio.Textbox(lines=1, placeholder=None, value="0", label='No. of siblings/spouse of the passenger aboard')
in_parch = gradio.Textbox(lines=1, placeholder=None, value="2", label='No. of parents/children of the passenger aboard')
in_ticket = gradio.Textbox(lines=1, placeholder=None, value="248738", label='Ticket number')
in_cabin = gradio.Textbox(lines=1, placeholder=None, value="A5", label='Cabin number')
in_embarked = gradio.Radio(["Southampton", "Cherbourg", "Queenstown"], type="value", label='Port of Embarkation')
in_fare = gradio.Textbox(lines=1, placeholder=None, value="29", label='Passenger fare')

# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")

# Label prediction function
def get_output_label(in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare):
    
    input_df = pd.DataFrame({"PassengerId": [in_Pid], 
                             "Pclass": [int(in_Pclass)], 
                             "Name": [in_Pname],
                             "Sex": [in_sex.lower()], 
                             "Age": [float(in_age)], 
                             "SibSp": [int(in_sibsp)],
                             "Parch": [int(in_parch)], 
                             "Ticket": [in_ticket], 
                             "Cabin": [in_cabin],
                             "Embarked": [in_embarked[0]], 
                             "Fare": [float(in_fare)]})
    
    result = make_prediction(input_data=input_df.replace({np.nan: None}))["predictions"]
    label = "Survived" if result[0]==1 else "Not Survived"
    return label


# Create Gradio interface object
iface = gradio.Interface(fn = get_output_label,
                         inputs = [in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare],
                         outputs = [out_label],
                         title="Titanic Survival Prediction API 2.0  ⛴",
                         description="Predictive model that answers the question: “What sort of people were more likely to survive?”",
                         allow_flagging='never',
                         )

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
