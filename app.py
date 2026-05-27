from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import io
import base64
from matplotlib import pyplot as plt

from visualizations import DashboardVisualizer

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# =========================================================
# DASHBOARD ENGINE
# =========================================================

try:
    dashboard_engine = DashboardVisualizer()
    print("Dashboard visualizer initialized successfully.")
except Exception as e:
    print(f"Error initializing dashboard visualizer: {e}")
    dashboard_engine = None

# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = pickle.load(open('robust_sentiment_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    print("Model and TF-IDF Vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit()

# =========================================================
# NLTK SETUP
# =========================================================

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
important_words = ['not', 'no', 'bad', 'worst', 'terrible']
all_stopwords = [word for word in all_stopwords if word not in important_words]

def preprocess_text_for_prediction(text):
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    processed_words = [ps.stem(word) for word in words if word not in all_stopwords]
    return ' '.join(processed_words)

# =========================================================
# ROUTES
# =========================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if dashboard_engine is None:
        flash("Dashboard is currently unavailable.", "error")
        return redirect(url_for('home'))

    # Generate interactive charts HTML
    pie_chart = dashboard_engine.sentiment_distribution()
    trend_chart = dashboard_engine.sentiment_trend()
    histogram_chart = dashboard_engine.rating_histogram()
    scatter_chart = dashboard_engine.rating_scatter()
    kmeans_chart = dashboard_engine.kmeans_clustering()

    # Generate static images (saves to disk)
    dashboard_engine.generate_wordcloud('Positive')
    dashboard_engine.generate_wordcloud('Neutral')
    dashboard_engine.generate_wordcloud('Negative')
    dashboard_engine.confusion_matrix_chart()
    dashboard_engine.correlation_matrix_chart()

    kpis = dashboard_engine.get_kpis()

    return render_template(
        'dashboard.html',
        kpis=kpis,
        pie_chart=pie_chart,
        trend_chart=trend_chart,
        histogram_chart=histogram_chart,
        scatter_chart=scatter_chart,
        kmeans_chart=kmeans_chart
    )

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        try:
            df_upload = pd.read_csv(file)
            review_column = 'verified_reviews' if 'verified_reviews' in df_upload.columns else df_upload.columns[0]
            predictions_list = []
            for index, row in df_upload.iterrows():
                review = row[review_column]
                processed_review = preprocess_text_for_prediction(review)
                vectorized_review = vectorizer.transform([processed_review])
                prediction = model.predict(vectorized_review)[0]
                sentiment = "Positive" if prediction == 1 else "Negative"
                predictions_list.append({'review': review, 'sentiment': sentiment})

            sentiment_counts = pd.Series([p['sentiment'] for p in predictions_list]).value_counts()
            fig, ax = plt.subplots(figsize=(6,6))
            sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
            ax.set_ylabel('')
            ax.set_title('Sentiment Distribution')
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            return render_template('results.html', predictions=predictions_list, plot_url=plot_url)
        except Exception as e:
            flash(f"Error processing file: {e}", 'error')
            return redirect(url_for('home'))
    else:
        review_text = request.form['review_text']
        processed_review = preprocess_text_for_prediction(review_text)
        vectorized_review = vectorizer.transform([processed_review])
        prediction = model.predict(vectorized_review)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        return render_template('results.html', single_prediction={'review': review_text, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)