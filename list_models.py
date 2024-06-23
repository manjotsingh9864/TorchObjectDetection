import os
from ultralytics import YOLO

# Function to list available models
def list_models():
    models_dir = os.path.dirname(__file__)
    models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    return models

# Print all available models
available_models = list_models()
print("Available YOLO models:", available_models)