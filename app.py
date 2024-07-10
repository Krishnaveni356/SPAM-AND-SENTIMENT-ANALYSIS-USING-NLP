from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secret key
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded images

# In-memory database for user profiles and query history
users_db = {
    'user1': {'password': 'password1', 'queries': []},
    'user2': {'password': 'password2', 'queries': []}
}

# Load the spam classification model and CountVectorizer
df = pd.read_csv("spam.csv", encoding="latin-1")
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# OCR function to extract text from an image
def ocr(image_path):
    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(image)
    return text

# Spam classification function
def classify_spam(text):
    data = [text]
    vect = cv.transform(data).toarray()
    spam_prediction = clf.predict(vect)
    return spam_prediction[0]

# Sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]

        # Sentiment Analysis
        blob = TextBlob(message)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Spam Classification
        vect = cv.transform(data).toarray()
        spam_prediction = clf.predict(vect)

        # Store the user's query in their profile
        if 'username' in session:
            username = session['username']
            users_db[username]['queries'].append(message)

        return render_template('result.html', sentiment=polarity, subjectivity=subjectivity, spam_prediction=spam_prediction[0])

@app.route('/ocr', methods=['POST'])
def ocr_process():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file:
            # Securely save the uploaded image
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Perform OCR on the uploaded image
            extracted_text = ocr(image_path)

            # Remove the temporary image file
            os.remove(image_path)

            # Classify the extracted text as spam or not
            spam_prediction = classify_spam(extracted_text)

            # Perform sentiment analysis on the extracted text
            polarity, subjectivity = analyze_sentiment(extracted_text)

            return render_template('ocr_result.html', extracted_text=extracted_text, spam_prediction=spam_prediction, sentiment_polarity=polarity, sentiment_subjectivity=subjectivity)

if __name__ == '__main__':
    app.run(debug=True)