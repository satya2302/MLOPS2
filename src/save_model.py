def save_model(model, filepath="fashion_mnist_model.h5"):
    model.save(filepath)
    print(f"Model saved as {filepath}")
