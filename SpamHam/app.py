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

# Home route for input form
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']  # Get message from form
        message_vectorized = vectorizer.transform([message])  # Vectorize the message
        prediction = model.predict(message_vectorized)  # Predict

        # Render result page
        return render_template('index.html', prediction=prediction[0], message=message)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
