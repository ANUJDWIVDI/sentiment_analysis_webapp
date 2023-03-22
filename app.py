import base64
import os

import nltk
from flask import Flask, request, make_response, render_template
import pandas as pd
import pdfkit
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import io
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('vader_lexicon')



app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('sentiment_analysis.html')


@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    try:
        print("ENTER -- TRY -- '/sentiment-analysis")
        text_col = request.args.get('text_col', 'review')
        label_col = request.args.get('label_col', 'sentiment')
        df = pd.read_csv('static/files/IMDB Dataset.csv')

        print("START -- Data preprocessing with NLTK")
        # Data preprocessing with NLTK
        stop_words = set(stopwords.words('english'))
        sid = SentimentIntensityAnalyzer()
        df['sentiment'] = df[text_col].apply(lambda x: sid.polarity_scores(x)['compound'])
        df[text_col] = df[text_col].apply(lambda x: ' '.join(
            [word for word in word_tokenize(x.lower()) if word.isalpha() and word not in stop_words]))
        print("END -- Data preprocessing with NLTK")


        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df[text_col], df[label_col], test_size=0.2, random_state=42)

        # Vectorize the text data
        vectorizer = CountVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        print("VECTORIZE DATA CHECK - ||| -")

        print("ENTER TRAIN MULTIMODAL NAIVE BAYES CLASSIFIER")
        # Train a Multinomial Naive Bayes classifier
        clf = MultinomialNB()
        print("FIT ENTER -- MULTIMODAL NAIVE BAYES CLASSIFIER --")
        clf.fit(X_train_vectorized, y_train)
        print("Model fit. check")
        y_pred = clf.predict(X_test_vectorized)
        print("Model predict check")


        # Calculate the accuracy and print the classification report
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy_score check")
        report = classification_report(y_test, y_pred)

        matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix Formation check")
        print("END TRAIN MULTIMODAL NAIVE BAYES CLASSIFIER")

        print("ENTER Visualizing the confusion matrix")
        # Visualize the confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(x=j, y=i, s=matrix[i, j], va='center', ha='center')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        image = io.BytesIO()
        plt.savefig(image, format='png')
        image.seek(0)
        print("END Visualizing the confusion matrix")

        print("START GENERATION OF REPORT - pDF")
        # Generate a PDF report and save it locally
        html = f'<h1>Classification Report:</h1><br><pre>{report}</pre><br><img src="data:image/png;base64,{base64.b64encode(image.read()).decode()}">'
        pdf = pdfkit.from_string(html, False)
        with open(os.path.join(app.static_folder, 'files', 'report.pdf'), 'wb') as f:
            f.write(pdf)
        print("END GENERATION OF REPORT - pDF")

        print("END '/sentiment-analysis'")
        # Render the result template with the path to the saved PDF file
        return render_template('result.html')


    except Exception as e:
        print("exception", e)
        return render_template('error.html')

if __name__ == '__main__':
    app.run()
