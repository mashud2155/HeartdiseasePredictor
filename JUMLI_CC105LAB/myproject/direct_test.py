"""
Direct test of model calibration
"""
import os
import pickle
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# Path to the model file
MODEL_PATH = os.path.join('myapp', 'data', 'heart_disease_model_jumli.pkl')

def main():
    # Load the model
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    
    print("Model loaded successfully")
    print(f"Model type: {type(model)}")
    print(f"Model keys: {model.keys()}")
    
    # Sample inputs
    high_risk_input = [
        [68, 175, 340, 98, 3.4, 
         0, 0, 1,  # rest_ecg: ST-T wave
         1, 1, 0,  # slope: flat, not upsloping
         0, 1, 0, 0,  # vessels: three
         0, 0, 1]   # thalassemia: reversable defect
    ]
    
    low_risk_input = [
        [40, 120, 180, 180, 0.1, 
         1, 1, 0,  # rest_ecg: normal
         0, 0, 1,  # slope: upsloping
         0, 0, 0, 1,  # vessels: zero
         0, 1, 0]   # thalassemia: normal
    ]
    
    # Check original predictions
    print("\nOriginal model predictions:")
    print_predictions(model['model'], high_risk_input, "High Risk")
    print_predictions(model['model'], low_risk_input, "Low Risk")
    
    # Perform manual calibration
    print("\nPerforming manual calibration...")
    calibrated_model = calibrate_model(model['model'])
    
    # Check calibrated predictions
    print("\nCalibrated model predictions:")
    print_predictions(calibrated_model, high_risk_input, "High Risk")
    print_predictions(calibrated_model, low_risk_input, "Low Risk")
    
    # Manual probability adjustment using sigmoid function
    print("\nManually adjusted probabilities using sigmoid:")
    for name, input_data in [("High Risk", high_risk_input), ("Low Risk", low_risk_input)]:
        raw_decision = model['model'].decision_function(input_data)[0]
        # Apply sigmoid transformation manually
        adjusted_prob = 1 / (1 + np.exp(-raw_decision))
        print(f"{name}: Raw decision value: {raw_decision}, Adjusted probability: {adjusted_prob:.4f}")

def print_predictions(model, input_data, name):
    """Print predictions for a given input"""
    prediction = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)[0]
    decision = getattr(model, 'decision_function', lambda x: None)(input_data)
    
    print(f"{name}:")
    print(f"  Prediction: {prediction} (Heart Disease: {'Yes' if prediction == 1 else 'No'})")
    print(f"  Probabilities: Negative: {probas[0]:.4f} ({probas[0]*100:.2f}%), Positive: {probas[1]:.4f} ({probas[1]*100:.2f}%)")
    if decision is not None:
        print(f"  Decision function value: {decision[0]:.4f}")

def calibrate_model(base_model):
    """Calibrate the model using Platt scaling for better probability estimates"""
    # Create a simple dataset for calibration
    np.random.seed(42)
    n_samples = 100
    
    # Generate random data within typical ranges
    age = np.random.uniform(30, 80, n_samples)
    resting_bp = np.random.uniform(90, 200, n_samples)
    cholestoral = np.random.uniform(120, 400, n_samples)
    max_heart_rate = np.random.uniform(70, 220, n_samples)
    oldpeak = np.random.uniform(0, 5, n_samples)
    
    # For binary features, generate 0 or 1
    binary_features = np.random.randint(0, 2, (n_samples, 13))
    
    # Combine all features
    X_calib = np.column_stack([age, resting_bp, cholestoral, max_heart_rate, oldpeak, binary_features])
    
    # Get predictions from the base model for these samples
    y_calib = base_model.predict(X_calib)
    
    # Create and fit a calibrated model - using 'estimator' parameter for newer sklearn versions
    try:
        # Try new parameter name first
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')
        calibrated_model.fit(X_calib, y_calib)
    except TypeError:
        # Fall back to old parameter name
        calibrated_model = CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv='prefit')
        calibrated_model.fit(X_calib, y_calib)
    
    return calibrated_model

if __name__ == "__main__":
    main() 