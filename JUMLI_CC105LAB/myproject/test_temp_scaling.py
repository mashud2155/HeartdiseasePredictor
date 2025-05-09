"""
Test script for temperature scaling approach
"""
import os
import pickle
import numpy as np

# Path to the model file
MODEL_PATH = os.path.join('myapp', 'data', 'heart_disease_model_jumli.pkl')

def temperature_scale(decision_value, temperature=10.0):
    """Apply temperature scaling to logits"""
    return 1.0 / (1.0 + np.exp(-decision_value / temperature))

def main():
    # Load the model
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    
    print("Model loaded successfully")
    
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
    
    # Test different temperature values
    temperatures = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    print("\n===== TEMPERATURE SCALING RESULTS =====")
    print("{:<12} {:<25} {:<25}".format("Temperature", "High Risk Probability", "Low Risk Probability"))
    print("-" * 65)
    
    # Original probabilities
    high_risk_proba = model['model'].predict_proba(high_risk_input)[0][1]
    low_risk_proba = model['model'].predict_proba(low_risk_input)[0][1]
    print("{:<12} {:<25} {:<25}".format(
        "Original", 
        f"{high_risk_proba:.4f} ({high_risk_proba*100:.2f}%)", 
        f"{low_risk_proba:.4f} ({low_risk_proba*100:.2f}%)"
    ))
    
    # Get raw decision values
    high_risk_decision = model['model'].decision_function(high_risk_input)[0]
    low_risk_decision = model['model'].decision_function(low_risk_input)[0]
    
    print("\nRaw decision values:")
    print(f"High Risk: {high_risk_decision:.4f}")
    print(f"Low Risk: {low_risk_decision:.4f}")
    print("-" * 65)
    
    # Test each temperature
    for temp in temperatures:
        # Calculate new probabilities
        high_risk_prob = temperature_scale(high_risk_decision, temp)
        low_risk_prob = temperature_scale(low_risk_decision, temp)
        
        print("{:<12} {:<25} {:<25}".format(
            f"T={temp}", 
            f"{high_risk_prob:.4f} ({high_risk_prob*100:.2f}%)", 
            f"{low_risk_prob:.4f} ({low_risk_prob*100:.2f}%)"
        ))

if __name__ == "__main__":
    main() 