import requests

def fetch_available_models():
    url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/models/yolov5s.yaml"
    response = requests.get(url)
    if response.status_code == 200:
        print("Available YOLO models for download:")
        print(response.text)
    else:
        print("Failed to fetch the list of models. Please check the URL or your internet connection.")

# Fetch and print the list of available models
fetch_available_models()
