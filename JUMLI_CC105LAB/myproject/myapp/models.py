from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    result = models.BooleanField()  # 1 for positive, 0 for negative
    probability = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Store prediction inputs
    age = models.FloatField()
    resting_bp = models.FloatField()
    cholestoral = models.FloatField()
    max_heart_rate = models.FloatField()
    oldpeak = models.FloatField()
    fasting_blood_sugar = models.BooleanField()
    rest_ecg = models.CharField(max_length=20)
    exercise_angina = models.BooleanField()
    slope = models.CharField(max_length=20)
    vessels_colored = models.CharField(max_length=20)
    thalassemia = models.CharField(max_length=20)
    
    def __str__(self):
        return f"Prediction for {self.user.username}: {'Positive' if self.result else 'Negative'}"
