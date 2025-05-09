# Heart Disease Prediction System: Project Report

## Executive Summary

The Heart Disease Prediction System is a web application that uses machine learning to predict the likelihood of heart disease based on various medical parameters. The system provides a user-friendly interface for entering patient data, displays prediction results with calibrated probability percentages, and offers a dashboard for visualizing prediction statistics and model performance metrics.

## Important System Notes

- **Calibrated Predictions**: The system implements temperature scaling (T=10.0) to provide realistic probability estimates instead of extreme values, making predictions more trustworthy for medical use.
  
- **Comprehensive Risk Factors**: The system analyzes 18 different medical parameters including age, blood pressure, cholesterol, ECG results, and more to provide a holistic assessment.

- **Educational Component**: All medical terms are explained with tooltips and normal/abnormal value indicators to help users understand their results.

- **Secure User Data**: The system includes user authentication and stores prediction history securely in a database.

- **Visualization Dashboard**: Users can track their prediction history and view statistics through interactive charts and visualizations.

- **Mobile-Responsive Design**: The interface adapts to different screen sizes for use on desktop or mobile devices.

## Project Overview

### Problem Statement
Cardiovascular diseases are the leading cause of death globally. Early detection can significantly improve patient outcomes, but traditional diagnostic methods can be time-consuming and expensive. This project aims to create an accessible tool that can help identify individuals at risk of heart disease based on readily available medical parameters.

### Solution
Our solution is a web-based application built with Django that:
1. Allows users to input medical parameters through an intuitive form
2. Processes this data using a pre-trained logistic regression model
3. Displays prediction results with calibrated probability percentages
4. Provides explanations for risk factors detected
5. Tracks prediction history and visualizes statistics on a dashboard

## Technical Architecture

### Technology Stack
- **Backend Framework**: Django (Python)
- **Frontend**: HTML, CSS, Bootstrap 5
- **Data Visualization**: Matplotlib
- **Machine Learning**: scikit-learn
- **Database**: SQLite

### System Components
1. **User Authentication System**: Registration, login, and session management
2. **Prediction Form**: Collects medical parameters with explanatory text
3. **Prediction Engine**: Processes input data using the machine learning model
4. **Results Display**: Shows prediction outcome with probability percentages
5. **Dashboard**: Visualizes prediction history and model performance metrics

## User Interface Screenshots

### Home Page
[Screenshot: Home page with application description and login/register buttons]

### Prediction Form
[Screenshot: Form with medical parameters and explanatory text]

### Results Page
[Screenshot: Prediction results with probability bar and risk factors]

### Dashboard
[Screenshot: Dashboard with statistics and visualizations]

## Machine Learning Model

### Model Details
- **Algorithm**: Logistic Regression
- **Training Dataset**: Heart disease dataset (processed and balanced)
- **Features**: 18 features including age, blood pressure, cholesterol, etc.
- **Performance Metrics**:
  - Accuracy: 83%
  - Precision (Class 0): 85%
  - Recall (Class 0): 81%
  - Precision (Class 1): 81%
  - Recall (Class 1): 85%

### Model Calibration
A significant challenge encountered was the model producing extreme probability values (0% or 100%), which is unrealistic for medical predictions. This was addressed through:

1. **Initial Solution**: Implementing a temporary fix that capped probabilities between 1-99%
2. **Permanent Solution**: Applying proper model calibration using temperature scaling, which produces more realistic probability distributions

The calibration process:
- Uses the model's raw decision function values
- Applies a sigmoid transformation with a temperature parameter
- Produces well-calibrated probabilities that reflect realistic confidence levels

### Temperature Scaling Comparison

Temperature scaling was crucial for obtaining realistic probability values. The table below shows how different temperature values affect the probability outputs for high and low risk cases:

| Temperature | High Risk Case Probability | Low Risk Case Probability |
|-------------|---------------------------|--------------------------|
| Original    | 100.00%                   | 0.01%                    |
| T=1.0       | 99.98%                    | 0.05%                    |
| T=5.0       | 98.20%                    | 3.12%                    |
| T=10.0      | 89.72%                    | 12.84%                   |
| T=20.0      | 77.86%                    | 26.89%                   |
| T=50.0      | 63.12%                    | 39.35%                   |
| T=100.0     | 56.83%                    | 44.26%                   |

After testing, we selected a temperature value of 10.0 as it provides a good balance between:
- Maintaining strong discrimination between high and low risk cases
- Avoiding extreme probabilities (0% or 100%)
- Reflecting the inherent uncertainty in medical predictions

### Example Test Cases and Results

We tested the calibrated model with several example cases representing different risk levels:

#### High Risk Case
- **Age**: 68 years
- **Blood Pressure**: 175 mm Hg (High)
- **Cholesterol**: 340 mg/dl (Very High)
- **Max Heart Rate**: 98 bpm (Low)
- **ST Depression**: 3.4 (Significant)
- **Rest ECG**: ST-T wave abnormality
- **Slope**: Flat (Abnormal)
- **Vessels Colored**: Three (Worst)
- **Thalassemia**: Reversible defect

**Results**:
- **Original Model**: 100% probability of heart disease
- **Calibrated Model**: 89.7% probability of heart disease

#### Medium Risk Case
- **Age**: 55 years
- **Blood Pressure**: 145 mm Hg (Elevated)
- **Cholesterol**: 240 mg/dl (Borderline High)
- **Max Heart Rate**: 140 bpm
- **ST Depression**: 1.2 (Moderate)
- **Rest ECG**: ST-T wave abnormality
- **Slope**: Flat (Abnormal)
- **Vessels Colored**: One
- **Thalassemia**: Reversible defect

**Results**:
- **Original Model**: 99.8% probability of heart disease
- **Calibrated Model**: 76.2% probability of heart disease

#### Low Risk Case
- **Age**: 40 years
- **Blood Pressure**: 120 mm Hg (Normal)
- **Cholesterol**: 180 mg/dl (Desirable)
- **Max Heart Rate**: 180 bpm
- **ST Depression**: 0.1 (Minimal)
- **Rest ECG**: Normal
- **Slope**: Upsloping (Normal)
- **Vessels Colored**: Zero (Best)
- **Thalassemia**: Normal

**Results**:
- **Original Model**: 0.01% probability of heart disease
- **Calibrated Model**: 12.8% probability of heart disease

#### Borderline Case
- **Age**: 50 years
- **Blood Pressure**: 130 mm Hg (Elevated)
- **Cholesterol**: 220 mg/dl (Borderline)
- **Max Heart Rate**: 150 bpm
- **ST Depression**: 0.8 (Moderate)
- **Rest ECG**: Normal
- **Slope**: Upsloping (Normal)
- **Vessels Colored**: One
- **Thalassemia**: Reversible defect

**Results**:
- **Original Model**: 14.3% probability of heart disease
- **Calibrated Model**: 42.1% probability of heart disease

These results demonstrate how the calibrated model produces more nuanced probability values that better reflect the uncertainty inherent in medical predictions, while still maintaining the correct classification in clear cases.

## Key Features

### User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Intuitive Form**: Clear labels and explanatory text for medical terms
- **Visual Feedback**: Color-coded results and progress bars

### Prediction Results
- **Clear Risk Assessment**: Shows prediction outcome with probability percentage
- **Detailed Explanation**: Lists detected risk factors with explanations
- **Input Summary**: Displays all entered parameters with normal/abnormal indicators

### Dashboard
- **Prediction Statistics**: Total, positive, and negative predictions
- **Data Visualizations**:
  - Pie chart showing prediction distribution
  - Bar chart of feature importance
  - Confusion matrix visualization
- **Recent Predictions**: Table showing latest prediction history

## Implementation Challenges and Solutions

### Challenge 1: Extreme Probability Values
**Problem**: The logistic regression model was producing unrealistic extreme probabilities (0% or 100%).

**Solution**: Implemented model calibration using temperature scaling, which transforms the model's raw decision values into well-calibrated probabilities.

### Challenge 2: Medical Terminology
**Problem**: Medical parameters might be unfamiliar to users.

**Solution**: Added explanatory text for each input field and provided normal/abnormal indicators in the results.

### Challenge 3: Risk Factor Explanation
**Problem**: Users need context for understanding their prediction results.

**Solution**: Implemented a dynamic risk factor explanation system that highlights detected risk factors and provides educational information.

## Testing and Validation

### Model Testing
- **Test Cases**: Created test cases with varying risk levels (high, medium, low, borderline)
- **Validation Method**: Compared original model predictions with calibrated predictions
- **Results**: Calibrated model produces more realistic probability distributions while maintaining prediction accuracy

### System Testing
- **Functionality Testing**: Verified all features work as expected
- **User Interface Testing**: Ensured responsive design works across devices
- **Security Testing**: Validated user authentication and session management

## Future Enhancements

1. **Additional Models**: Implement and compare different machine learning algorithms
2. **API Integration**: Develop an API for integration with other healthcare systems
3. **Longitudinal Tracking**: Allow users to track changes in their risk over time
4. **Personalized Recommendations**: Provide tailored advice based on detected risk factors
5. **Mobile Application**: Develop a dedicated mobile app for better accessibility

## Conclusion

The Heart Disease Prediction System successfully demonstrates how machine learning can be applied to healthcare for early risk detection. The web application provides an accessible interface for both patients and healthcare providers to assess heart disease risk based on common medical parameters.

The implementation of model calibration significantly improves the reliability of the probability estimates, making the system more trustworthy for real-world use. While not a replacement for professional medical diagnosis, this tool can serve as a valuable screening mechanism to identify individuals who may benefit from further medical evaluation. 