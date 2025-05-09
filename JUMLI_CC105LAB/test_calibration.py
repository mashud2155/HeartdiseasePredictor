import os
import sys
import numpy as np

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.myproject.settings')

# Initialize Django
import django
django.setup()

# Import after Django setup
from myproject.myapp.views import model, calibrated_model

# Test cases with different risk levels
def test_calibration():
    # Sample inputs with varying risk levels
    test_cases = [
        # High risk case
        {
            "name": "High Risk",
            "input": [
                [68, 175, 340, 98, 3.4, 
                0, 0, 1,  # rest_ecg: ST-T wave
                1, 1, 0,  # slope: flat, not upsloping
                0, 1, 0, 0,  # vessels: three
                0, 0, 1]   # thalassemia: reversable defect
            ]
        },
        # Medium risk case
        {
            "name": "Medium Risk",
            "input": [
                [55, 145, 240, 140, 1.2, 
                0, 0, 1,  # rest_ecg: ST-T wave
                1, 1, 0,  # slope: flat
                1, 0, 0, 0,  # vessels: one
                0, 0, 1]   # thalassemia: reversable defect
            ]
        },
        # Low risk case
        {
            "name": "Low Risk",
            "input": [
                [40, 120, 180, 180, 0.1, 
                1, 1, 0,  # rest_ecg: normal
                0, 0, 1,  # slope: upsloping
                0, 0, 0, 1,  # vessels: zero
                0, 1, 0]   # thalassemia: normal
            ]
        },
        # Borderline case
        {
            "name": "Borderline Case",
            "input": [
                [50, 130, 220, 150, 0.8, 
                0, 1, 0,  # rest_ecg: normal
                0, 0, 1,  # slope: upsloping
                1, 0, 0, 0,  # vessels: one
                0, 0, 1]   # thalassemia: reversable defect
            ]
        }
    ]
    
    print("\n===== TESTING MODEL CALIBRATION =====")
    print(f"{'Case Name':<15} {'Original Prediction':<20} {'Original Probability':<25} {'Calibrated Prediction':<25} {'Calibrated Probability':<25}")
    print("-" * 110)
    
    for case in test_cases:
        # Get original model predictions
        orig_pred = model['model'].predict(case['input'])[0]
        orig_proba = model['model'].predict_proba(case['input'])[0]
        orig_proba_percent = [round(p * 100, 2) for p in orig_proba]
        
        # Create prediction text
        orig_pred_str = "Heart Disease" if orig_pred == 1 else "No Heart Disease"
        
        # Get calibrated model predictions
        if calibrated_model is not None:
            calib_proba = calibrated_model.predict_proba(case['input'])[0]
            calib_pred = 1 if calib_proba[1] >= 0.5 else 0
            calib_proba_percent = [round(p * 100, 2) for p in calib_proba]
            calib_pred_str = "Heart Disease" if calib_pred == 1 else "No Heart Disease"
        else:
            calib_pred = "N/A"
            calib_proba_percent = ["N/A", "N/A"]
            calib_pred_str = "N/A"
        
        # Print results
        print(f"{case['name']:<15} {orig_pred} ({orig_pred_str:<15}) {orig_proba_percent[0]}% / {orig_proba_percent[1]}%{' ':10} {calib_pred} ({calib_pred_str:<15}) {calib_proba_percent[0]}% / {calib_proba_percent[1]}%")

if __name__ == "__main__":
    test_calibration() 