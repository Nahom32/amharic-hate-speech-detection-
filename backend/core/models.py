from django.db import models

# Create your models here.
class UserInput(models.Model):
    text = models.TextField()
    predicted_class = models.CharField(max_length=20)
    prediction_confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text