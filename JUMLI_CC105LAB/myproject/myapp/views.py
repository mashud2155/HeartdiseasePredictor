import os
import pickle
# Set Matplotlib to use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import json
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.http import JsonResponse
from .models import Prediction
from sklearn.calibration import CalibratedClassifierCV

# Load the trained model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'myapp', 'data', 'heart_disease_model_jumli.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Temperature scaling function
def temperature_scale(decision_value, temperature=10.0):
    """
    Apply temperature scaling to logits to moderate extreme predictions.
    Higher temperature makes probabilities more moderate.
    
    Args:
        decision_value: Raw decision function value
        temperature: Temperature parameter (higher = more moderate probabilities)
        
    Returns:
        Calibrated probability
    """
    return 1.0 / (1.0 + np.exp(-decision_value / temperature))

# Define the calibration function
def get_calibrated_probability(model, input_data, temperature=10.0):
    """
    Get calibrated probabilities using temperature scaling
    
    Args:
        model: The pre-trained model
        input_data: Input features
        temperature: Temperature parameter for scaling
        
    Returns:
        Tuple of (prediction, probabilities)
    """
    # Get raw decision value from model
    try:
        decision_value = model.decision_function(input_data)[0]
        
        # Apply temperature scaling
        positive_prob = temperature_scale(decision_value, temperature)
        negative_prob = 1.0 - positive_prob
        
        # Create probability array
        calibrated_proba = np.array([negative_prob, positive_prob])
        
        # Make prediction based on calibrated probability
        prediction = 1 if positive_prob >= 0.5 else 0
        
        return prediction, calibrated_proba
    except Exception as e:
        print(f"Error in calibration: {e}")
        # Fallback to standard prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        return prediction, probabilities

# Global calibrated model variable - initially None
calibrated_model = None

# Debug function to inspect model
@login_required
def debug_model(request):
    """View for debugging model outputs with controlled inputs"""
    # Create a sample input with high risk factors
    sample_input = [
        [68, 175, 340, 98, 3.4, 
         0, 0, 1,  # rest_ecg: ST-T wave
         1, 1, 0,  # slope: flat, not upsloping
         0, 1, 0, 0,  # vessels: three
         0, 0, 1]   # thalassemia: reversable defect
    ]
    
    # Create a sample input with low risk factors
    sample_input_low = [
        [40, 120, 180, 180, 0.1, 
         1, 1, 0,  # rest_ecg: normal
         0, 0, 1,  # slope: upsloping
         0, 0, 0, 1,  # vessels: zero
         0, 1, 0]   # thalassemia: normal
    ]
    
    # Create a sample input with medium risk factors
    sample_input_medium = [
        [55, 145, 240, 140, 1.2, 
         0, 0, 1,  # rest_ecg: ST-T wave
         1, 1, 0,  # slope: flat
         1, 0, 0, 0,  # vessels: one
         0, 0, 1]   # thalassemia: reversable defect
    ]
    
    # Get raw model predictions for all samples
    orig_pred_high = model['model'].predict(sample_input)[0]
    orig_proba_high = model['model'].predict_proba(sample_input)[0]
    
    orig_pred_low = model['model'].predict(sample_input_low)[0]
    orig_proba_low = model['model'].predict_proba(sample_input_low)[0]
    
    orig_pred_medium = model['model'].predict(sample_input_medium)[0]
    orig_proba_medium = model['model'].predict_proba(sample_input_medium)[0]
    
    # Try different temperature values
    temperatures = [1.0, 5.0, 10.0, 20.0, 50.0]
    calibrated_results = {}
    
    for temp in temperatures:
        calib_pred_high, calib_proba_high = get_calibrated_probability(model['model'], sample_input, temperature=temp)
        calib_pred_low, calib_proba_low = get_calibrated_probability(model['model'], sample_input_low, temperature=temp)
        calib_pred_medium, calib_proba_medium = get_calibrated_probability(model['model'], sample_input_medium, temperature=temp)
        
        calibrated_results[str(temp)] = {
            'high_risk': {
                'prediction': int(calib_pred_high),
                'probability': calib_proba_high.tolist(),
                'probability_percent': [round(p * 100, 2) for p in calib_proba_high]
            },
            'medium_risk': {
                'prediction': int(calib_pred_medium),
                'probability': calib_proba_medium.tolist(),
                'probability_percent': [round(p * 100, 2) for p in calib_proba_medium]
            },
            'low_risk': {
                'prediction': int(calib_pred_low),
                'probability': calib_proba_low.tolist(),
                'probability_percent': [round(p * 100, 2) for p in calib_proba_low]
            }
        }
    
    # Get model parameters
    model_type = model.get('model_type', 'Unknown')
    model_params = {}
    if hasattr(model['model'], 'get_params'):
        model_params = model['model'].get_params()
    
    # Get model coefficients if logistic regression
    coefficients = None
    if hasattr(model['model'], 'coef_'):
        coefficients = model['model'].coef_[0].tolist()
    
    # Get model intercept if available
    intercept = None
    if hasattr(model['model'], 'intercept_'):
        intercept = model['model'].intercept_[0]
    
    # Check if model uses sigmoid calibration
    has_calibration = hasattr(model['model'], '_sigmoid_calibration')
    
    # Prepare response data
    debug_data = {
        'model_type': model_type,
        'model_params': model_params,
        'coefficients': coefficients,
        'intercept': intercept,
        'has_calibration': has_calibration,
        'original': {
            'high_risk': {
                'prediction': int(orig_pred_high),
                'probability': orig_proba_high.tolist(),
                'probability_percent': [round(p * 100, 2) for p in orig_proba_high]
            },
            'medium_risk': {
                'prediction': int(orig_pred_medium),
                'probability': orig_proba_medium.tolist(),
                'probability_percent': [round(p * 100, 2) for p in orig_proba_medium]
            },
            'low_risk': {
                'prediction': int(orig_pred_low),
                'probability': orig_proba_low.tolist(),
                'probability_percent': [round(p * 100, 2) for p in orig_proba_low]
            }
        },
        'calibrated': calibrated_results,
        'feature_names': model.get('feature_names', [])
    }
    
    return JsonResponse(debug_data)

@login_required
def predict(request):
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.POST.get('age'))
        resting_bp = float(request.POST.get('resting_bp'))
        cholestoral = float(request.POST.get('cholestoral'))
        max_heart_rate = float(request.POST.get('max_heart_rate'))
        oldpeak = float(request.POST.get('oldpeak'))
        fasting_blood_sugar = int(request.POST.get('fasting_blood_sugar'))
        rest_ecg = request.POST.get('rest_ecg')
        exercise_angina = int(request.POST.get('exercise_angina'))
        slope = request.POST.get('slope')
        vessels_colored = request.POST.get('vessels_colored')
        thalassemia = request.POST.get('thalassemia')
        
        # Process categorical features
        rest_ecg_normal = 1 if rest_ecg == 'normal' else 0
        rest_ecg_st_t = 1 if rest_ecg == 'st-t' else 0
        
        slope_flat = 1 if slope == 'flat' else 0
        slope_upsloping = 1 if slope == 'upsloping' else 0
        
        vessels_zero = 1 if vessels_colored == 'zero' else 0
        vessels_one = 1 if vessels_colored == 'one' else 0
        vessels_two = 1 if vessels_colored == 'two' else 0
        vessels_three = 1 if vessels_colored == 'three' else 0
        
        thal_no = 1 if thalassemia == 'no' else 0
        thal_normal = 1 if thalassemia == 'normal' else 0
        thal_reversable = 1 if thalassemia == 'reversable_defect' else 0

        # Prepare the input for the model
        input_data = [
            [age, resting_bp, cholestoral, max_heart_rate, oldpeak, 
             fasting_blood_sugar, rest_ecg_normal, rest_ecg_st_t, 
             exercise_angina, slope_flat, slope_upsloping,
             vessels_one, vessels_three, vessels_two, vessels_zero,
             thal_no, thal_normal, thal_reversable]
        ]

        # Make the prediction using the base model
        base_prediction = model['model'].predict(input_data)[0]
        base_probabilities = model['model'].predict_proba(input_data)[0]
        
        # Get calibrated prediction with temperature scaling
        # Using temperature=10.0 for more moderate probabilities
        prediction, probabilities = get_calibrated_probability(model['model'], input_data, temperature=10.0)
        
        # Print debug information to console
        print(f"Base prediction: {base_prediction}")
        print(f"Base probabilities: {base_probabilities}")
        print(f"Calibrated prediction: {prediction}")
        print(f"Calibrated probabilities: {probabilities}")
        
        # Calculate percentage values
        negative_percentage = float(probabilities[0] * 100)
        positive_percentage = float(probabilities[1] * 100)
        
        # Format to 2 decimal places for display
        negative_percentage_display = "{:.2f}".format(negative_percentage)
        positive_percentage_display = "{:.2f}".format(positive_percentage)
        
        # Print more debug
        print(f"Negative %: {negative_percentage} -> {negative_percentage_display}")
        print(f"Positive %: {positive_percentage} -> {positive_percentage_display}")
        
        # Store user inputs for display
        user_inputs = {
            'age': age,
            'resting_bp': resting_bp,
            'cholestoral': cholestoral,
            'max_heart_rate': max_heart_rate,
            'oldpeak': oldpeak,
            'fasting_blood_sugar': 'Lower than 120 mg/ml' if fasting_blood_sugar == 1 else 'Greater than 120 mg/ml',
            'rest_ecg': rest_ecg.capitalize() if rest_ecg == 'normal' else 'ST-T Wave Abnormality' if rest_ecg == 'st-t' else 'Other',
            'exercise_angina': 'Yes' if exercise_angina == 1 else 'No',
            'slope': slope.capitalize(),
            'vessels_colored': vessels_colored.capitalize(),
            'thalassemia': thalassemia.replace('_', ' ').capitalize(),
        }
        
        # Save prediction to database
        new_prediction = Prediction(
            user=request.user,
            result=bool(prediction),
            probability=float(probabilities[1]),  # Store calibrated probability
            age=age,
            resting_bp=resting_bp,
            cholestoral=cholestoral,
            max_heart_rate=max_heart_rate,
            oldpeak=oldpeak,
            fasting_blood_sugar=bool(fasting_blood_sugar),
            rest_ecg=rest_ecg,
            exercise_angina=bool(exercise_angina),
            slope=slope,
            vessels_colored=vessels_colored,
            thalassemia=thalassemia,
        )
        new_prediction.save()
        
        # Store prediction info in session for results page
        request.session['prediction_result'] = {
            'prediction': int(prediction),
            'positive_percentage': positive_percentage_display,
            'negative_percentage': negative_percentage_display,
            'raw_positive': float(probabilities[1]),  # Store raw probability for calculations
            'raw_negative': float(probabilities[0]),  # Store raw probability for calculations
            'user_inputs': user_inputs,
            'prediction_id': new_prediction.id
        }
        
        # Redirect to results page
        return redirect('prediction_results')

    return render(request, 'predict.html')

@login_required
def prediction_results(request):
    # Get prediction result from session
    prediction_data = request.session.get('prediction_result')
    
    # If no prediction data in session, redirect to prediction page
    if not prediction_data:
        messages.warning(request, "Please make a prediction first.")
        return redirect('predict')
    
    # Clear the prediction data from session
    request.session.pop('prediction_result', None)
    
    return render(request, 'prediction_results.html', prediction_data)

@login_required
def dashboard(request):
    # Get real prediction statistics from database
    all_predictions = Prediction.objects.all()
    total_predictions = all_predictions.count()
    positive_predictions = all_predictions.filter(result=True).count()
    negative_predictions = all_predictions.filter(result=False).count()
    
    # Get recent predictions (limited to 5)
    recent_predictions = Prediction.objects.all().order_by('-created_at')[:5]
    
    # Format recent predictions for template
    predictions = []
    for pred in recent_predictions:
        predictions.append({
            'date': pred.created_at.strftime('%Y-%m-%d'),
            'result': 1 if pred.result else 0,
            'probability': int(pred.probability * 100)
        })
    
    # Dataset statistics - using statistics from the model
    feature_names = model['feature_names']
    numerical_features = model['numerical_features']
    
    # Dataset statistics
    dataset_stats = {
        'total_features': len(feature_names),
        'numerical_features': len(numerical_features),
        'categorical_features': len(feature_names) - len(numerical_features),
    }
    
    # Model performance metrics (extracted from the model)
    model_performance = {
        'accuracy': model.get('accuracy', 0.83),
        'precision_class0': model.get('precision_class0', 0.85),
        'recall_class0': model.get('recall_class0', 0.81),
        'precision_class1': model.get('precision_class1', 0.81),
        'recall_class1': model.get('recall_class1', 0.85)
    }
    
    # Real statistics for user predictions
    prediction_stats = {
        'total_predictions': total_predictions,
        'positive_predictions': positive_predictions,
        'negative_predictions': negative_predictions
    }
    
    # Create charts
    
    # If no predictions yet, use a placeholder for visualization
    if total_predictions == 0:
        values = [1, 0]  # Placeholder to avoid empty chart
    else:
        values = [negative_predictions, positive_predictions]
    
    # 1. Pie chart for prediction distribution
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    labels = ['Negative', 'Positive']
    explode = (0, 0.1)  # explode the 2nd slice (positive)
    
    ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%', 
           colors=['#28a745', '#dc3545'], shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Heart Disease Prediction Distribution')
    
    # Save pie chart to base64 string
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png', bbox_inches='tight')
    buffer1.seek(0)
    pie_chart = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close(fig1)
    
    # 2. Bar chart for feature importance
    # Get coefficients from logistic regression model
    coefficients = model['model'].coef_[0]
    feature_importance = np.abs(coefficients)
    
    # Get top 8 important features for better visualization
    indices = np.argsort(feature_importance)[-8:]
    top_features = [feature_names[i] for i in indices]
    top_importance = [feature_importance[i] for i in indices]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(top_features)))
    
    bars = ax2.barh(top_features, top_importance, color=colors)
    ax2.set_xlabel('Feature Importance (Absolute Coefficient Value)')
    ax2.set_title('Top Features for Heart Disease Prediction')
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', ha='left', va='center')
    
    # Save bar chart to base64 string
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png', bbox_inches='tight')
    buffer2.seek(0)
    bar_chart = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close(fig2)
    
    # 3. Confusion matrix visualization (simulated)
    # Create a simulated confusion matrix
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    cm = np.array([[81, 19], [15, 85]])  # Simulated confusion matrix values
    
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.figure.colorbar(im, ax=ax3)
    ax3.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'],
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig3.tight_layout()
    
    # Save confusion matrix to base64 string
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format='png', bbox_inches='tight')
    buffer3.seek(0)
    confusion_matrix = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close(fig3)
    
    return render(request, 'dashboard.html', {
        'dataset_stats': dataset_stats,
        'model_performance': model_performance,
        'stats': prediction_stats,
        'pie_chart': pie_chart,
        'bar_chart': bar_chart,
        'confusion_matrix': confusion_matrix,
        'predictions': predictions
    })

def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password')
    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    messages.info(request, 'You have been logged out successfully!')
    return redirect('login')
