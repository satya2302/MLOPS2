import yaml
from fetch_data import fetch_data
from preprocess_data import preprocess_data
from build_model import build_model
from train_model import train_model
from evaluate_model import evaluate_model
from save_model import save_model
from visualization import visualize_training
from explainability import explain_model

with open('pipeline_config.yml', 'r') as file:
    config = yaml.safe_load(file)

for step in config['pipeline']:
    print(f"Executing step: {step['name']} - {step['description']}")
    
    if step['step'] == "fetch_data":
        (train_images, train_labels), (test_images, test_labels) = fetch_data()

    elif step['step'] == "preprocess_data":
        X_train_scaled, X_test_scaled = preprocess_data(train_images, test_images)

    elif step['step'] == "build_model":
        model = build_model()

    elif step['step'] == "train_model":
        history = train_model(model, X_train_scaled, train_labels)

    elif step['step'] == "evaluate_model":
        evaluate_model(model, X_test_scaled, test_labels)

    elif step['step'] == "save_model":
        save_model(model)

    elif step['step'] == "visualization":
        visualize_training(history)

    elif step['step'] == "explainability":
        explain_model(model, X_train_scaled, X_test_scaled)
