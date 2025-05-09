# Heart Disease Prediction System
A machine learning-powered web application that predicts heart disease risk based on medical parameters. Built with Django and scikit-learn.

## Features
- **Heart Disease Risk Prediction**: Get personalized heart disease risk assessment based on your medical parameters
- **User Dashboard**: Track your predictions over time with visualizations and statistics
- **Calibrated Probabilities**: Reliable probability estimates using temperature scaling for more trustworthy results
- **Risk Factor Analysis**: Understand which factors contribute to your heart disease risk
- **Secure Authentication**: Create an account and securely manage your health data
- **Responsive Design**: Access the application from any device

## Demo
[Dashboard Screenshot]

## Installation
1. Clone the repository
```
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Run migrations
```
cd myproject
python manage.py migrate
```

5. Start the development server
```
python manage.py runserver
```

6. Visit http://127.0.0.1:8000/ in your browser

## Usage
1. Register an account or login if you already have one
2. Navigate to the Predict page from the dashboard
3. Fill in the form with your medical parameters
4. Submit the form to receive your heart disease risk prediction
5. View your results with detailed risk factor explanations
6. Track your prediction history on your dashboard

## Technologies Used
- **Backend**: Django (Python)
- **Frontend**: HTML, CSS, Bootstrap 5
- **Machine Learning**: scikit-learn (Logistic Regression with calibration)
- **Database**: SQLite (development), PostgreSQL (recommended for production)
- **Data Visualization**: Matplotlib
- **Data Processing**: pandas, numpy

## Project Structure
```
myproject/
├── myproject/           # Main Django project settings
├── myapp/               # Main application with views, models, forms
│   ├── data/            # Contains ML model
│   ├── templates/       # HTML templates
│   └── management/      # Custom management commands
└── manage.py            # Django command-line utility
```

## ML Model
The application uses a Logistic Regression model trained on heart disease data to predict risk levels. The model achieves 83% accuracy and has been calibrated using temperature scaling to provide realistic probability estimates.

The model evaluates 18 different medical parameters including:
- Age and blood pressure
- Cholesterol levels
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- Number of major vessels colored by fluoroscopy
- Thalassemia (blood disorder)

## Example Prediction Results

| Case Type | Example Profile | Original Model | Calibrated Model |
|-----------|----------------|----------------|------------------|
| High Risk | 68yr, high BP (175), cholesterol 340, 3 vessels | 100% | 89.7% |
| Medium Risk | 55yr, moderate BP (145), cholesterol 240, 1 vessel | 99.8% | 76.2% |
| Low Risk | 40yr, normal BP (120), cholesterol 180, 0 vessels | 0.01% | 12.8% |
| Borderline | 50yr, elevated BP (130), cholesterol 220, 1 vessel | 14.3% | 42.1% |

## Deployment
For production deployment, we recommend:
1. Using PostgreSQL as the database
2. Setting up proper environment variables
3. Configuring a production web server like Nginx with Gunicorn
4. Enabling HTTPS with Let's Encrypt

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
