from django.shortcuts import render
from django.http import JsonResponse
from .models import UserInput
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained hate speech detection model
model = tf.keras.models.load_model('path_to_your_model.h5')  # Replace with the actual model file path

# Preprocess the input text
def preprocess_text(text):
    # Add your text preprocessing code here (e.g., tokenization, padding, etc.)
    return preprocessed_text

def predict_hate_speech(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        preprocessed_text = preprocess_text(input_text)
        
        # Tokenize and pad the preprocessed text
        # (Assuming you have the tokenizer and max_len defined as in the previous code)
        X = tokenizer.texts_to_sequences([preprocessed_text])
        X = pad_sequences(X, maxlen=max_len)
        
        # Make a prediction using the loaded model
        prediction = model.predict(X)
        predicted_class = "Hate Speech" if prediction > 0.5 else "Not Hate Speech"
        
        # Save the user input and prediction result to the database
        user_input = UserInput.objects.create(text=input_text, predicted_class=predicted_class, prediction_confidence=float(prediction))
        
        return JsonResponse({'prediction': predicted_class, 'confidence': float(prediction)})
    return render(request, 'hate_speech_form.html')
