from django.urls import path
from .views import predict_hate_speech

urlpatterns = [
    path('predict_hate_speech/', predict_hate_speech, name='hate_speech_form'),
]