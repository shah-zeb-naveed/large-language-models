import os
import mlflow
from transformers import pipeline


# Use a pipeline as a high-level helper
# pipe = pipeline("text-generation", model="shahzebnaveed/codeparrot-ds")

# Construct and save the model
model_path = "codeparrotmlflow/"
transformers_model = CodeParrot()

artifacts = {
    "model": "codeparrot-ds/"
}

mlflow.pyfunc.save_model(
    path=model_path,
    python_model=transformers_model,
    artifacts=artifacts
)