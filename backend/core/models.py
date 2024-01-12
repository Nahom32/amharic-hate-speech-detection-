from django.db import models

class UserInput(models.Model):
    text = models.TextField()
    predicted_class = models.CharField(max_length=20)
    prediction_confidence = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)

    def str(self):
        return self.text