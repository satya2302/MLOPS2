def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    return history
