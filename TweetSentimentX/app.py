from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vector.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define label mapping
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Irrelevant"}

# Home route for input form
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if 'message' is part of the form
        if 'message' not in request.form:
            return "Error: Message key is missing from form data", 400
        
        message = request.form['message']  # Get message from form
        message_vectorized = vectorizer.transform([message])  # Vectorize the message
        prediction = model.predict(message_vectorized)  # Predict

        # Map the numerical prediction to the sentiment label
        predicted_label = label_mapping.get(prediction[0], "Unknown")

        # Render result page
        return render_template(
            'index.html',
            prediction=predicted_label,
            message=message
        )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
