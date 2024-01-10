from django.urls import path
from . import views

urlpatterns = [
    path('hate_speech_form/', views.hate_speech_form_view, name='hate_speech_form'),
    # Add other URL patterns for the core app here
]
