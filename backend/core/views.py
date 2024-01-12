import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from .models import UserInput
from joblib import load
from scipy.sparse import csr_matrix
import os
from django.conf import settings

# Custom pad_sequences function
def pad_sequences(sequences, maxlen, padding='post', value=0):
    # Creating a new zero-filled matrix
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            # Truncate the sequence if it is longer than maxlen
            padded_sequences[i] = np.array(seq[:maxlen])
        else:
            # Pad the sequence with the specified value
            if padding == 'post':
                padded_sequences[i, :len(seq)] = np.array(seq)
            else:  # 'pre' padding
                padded_sequences[i, -len(seq):] = np.array(seq)
    return padded_sequences

# Set the path to the model and tokenizer
model_path = os.path.join(settings.BASE_DIR,'models', 'random_forest_model.joblib')
tokenizer_path = os.path.join(settings.BASE_DIR,'models', 'tfidf_vectorizer.joblib')
# Load the pre-trained hate speech detection model and tokenizer
model = load(model_path)
tokenizer = load(tokenizer_path)

# Preprocess the input text
def preprocess_text(text):
    # Tokenize the text using your loaded tokenizer
    X = tokenizer.transform([text])
    X_dense = X.toarray() if isinstance(X, csr_matrix) else X
    # Pad the tokenized sequence to the desired length
    X = pad_sequences(X_dense, maxlen=2000)  
    return X

def predict_hate_speech(request):
    if request.method == 'POST':
        input_text = request.POST.get('inputText')
        preprocessed_text = preprocess_text(input_text)
        
        # Make a prediction using the loaded model
        prediction = model.predict(preprocessed_text)
        
        predicted_class = prediction[0]
        
        # Save the user input and prediction result to the database
        user_input = UserInput.objects.create(text=input_text, predicted_class=predicted_class, prediction_confidence=prediction[0])
        
        return JsonResponse({'prediction': predicted_class, 'confidence': prediction[0]})
    return render(request, 'hate_speech_form.html')