Python 3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> from flask import Flask, render_template, request, jsonify
... import joblib
... import nltk
... from nltk.corpus import stopwords
... from nltk.stem import WordNetLemmatizer
... import re
... 
... # Initialize Flask app
... app = Flask(__name__)
... 
... # Load the trained model and vectorizer
... model = joblib.load('fake_news_model.pkl')
... vectorizer = joblib.load('tfidf_vectorizer.pkl')
... 
... # Preprocessing function
... nltk.download('stopwords')
... nltk.download('wordnet')
... stop_words = set(stopwords.words('english'))
... lemmatizer = WordNetLemmatizer()
... 
... def preprocess_text(text):
...     text = text.lower()
...     text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
...     text = re.sub(r'\@w+|\#','', text)
...     text = re.sub(r'[^A-Za-z\s]', '', text)
...     words = text.split()
...     words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
...     return ' '.join(words)
... 
... @app.route('/')
... def home():
...     return render_template('index.html')
... 
... @app.route('/predict', methods=['POST'])
... def predict():
...     user_input = request.form['news_text']
...     processed_input = preprocess_text(user_input)
    transformed_input = vectorizer.transform([processed_input])
    prediction = model.predict(transformed_input)[0]
    label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
    return render_template('index.html', prediction=label, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
