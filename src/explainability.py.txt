import shap

def explain_model(model, X_train_scaled, X_test_scaled):
    explainer = shap.KernelExplainer(model.predict, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled[:10])
    shap.summary_plot(shap_values, X_test_scaled[:10])
