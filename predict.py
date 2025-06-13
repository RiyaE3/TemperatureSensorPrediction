import numpy as np

def predict_temperature(model, scaler, cell_voltages):
    features = np.array(cell_voltages).reshape(1, -1)
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    return predictions