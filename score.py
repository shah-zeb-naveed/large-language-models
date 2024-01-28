import os
import json
from azureml.studio.core.io.model_directory import ModelDirectory
import mlflow.pyfunc


def init():
    global model
    model = mlflow.pyfunc.load_model(
        os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'codeparrotmlflow/')
    )
    
def run(data):
    """
    Calls model.predict() over data and returns output in a dict.    
    """
    prediction = model.predict(data)
    # format for endpoint response
    return json.dumps(prediction.tolist())