import os
import re
import io
import base64
import pickle
import warnings

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.io as pio

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

warnings.filterwarnings('ignore')

# =========================================================
# NLTK SETUP
# =========================================================

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# =========================================================
# TEXT PREPROCESSING
# =========================================================

ps = PorterStemmer()

all_stopwords = stopwords.words('english')

important_words = [
    'not',
    'no',
    'bad',
    'worst',
    'terrible'
]

custom_stopwords = [
    w for w in all_stopwords
    if w not in important_words
]

def clean_text(text):
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    processed = [
        ps.stem(w)
        for w in words
        if w not in custom_stopwords
    ]
    return ' '.join(processed)

# =========================================================
# MAIN VISUALIZER CLASS
# =========================================================

class DashboardVisualizer:

    def __init__(self):
        self.save_dir = "static/visualizations"
        os.makedirs(self.save_dir, exist_ok=True)

        self.df = pd.read_csv("amazon_alexa.tsv", sep='\t')

        with open('robust_sentiment_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        with open('tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

        self.preprocess_data()

    def preprocess_data(self):
        self.df = self.df.dropna()
        self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
        self.df = self.df.dropna(subset=['rating'])

        def get_sentiment(rating):
            if rating >= 4:
                return "Positive"
            elif rating == 3:
                return "Neutral"
            else:
                return "Negative"

        self.df['sentiment'] = self.df['rating'].apply(get_sentiment)

        print("\n========== SENTIMENT COUNTS ==========\n")
        print(self.df['sentiment'].value_counts())
        print("\n======================================\n")

        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['review_length'] = self.df['verified_reviews'].apply(lambda x: len(str(x)))

        processed_texts = self.df['verified_reviews'].apply(clean_text)
        self.X_vectors = self.vectorizer.transform(processed_texts)
        self.df['predicted_feedback'] = self.model.predict(self.X_vectors)

    def get_kpis(self):
        total = len(self.df)
        positive = len(self.df[self.df['sentiment'] == 'Positive'])
        neutral = len(self.df[self.df['sentiment'] == 'Neutral'])
        negative = len(self.df[self.df['sentiment'] == 'Negative'])
        accuracy = accuracy_score(self.df['feedback'], self.df['predicted_feedback']) * 100
        return {
            'total': total,
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'accuracy': round(accuracy, 2)
        }

    # =====================================================
    # FIX PIE CHART
    # =====================================================
    def sentiment_distribution(self):
        sentiment_counts = self.df['sentiment'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            hole=0.4,
            title='Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive':'#48bb78',
                'Neutral':'#ecc94b',
                'Negative':'#f56565'
            }
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        fig.update_layout(
            autosize=True,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig.to_html(full_html=False)

    # =====================================================
    # FIX TREND CHART
    # =====================================================
    def sentiment_trend(self):
        trend = self.df.groupby(['date', 'sentiment']).size().reset_index(name='count')
        fig = px.line(
            trend,
            x='date',
            y='count',
            color='sentiment',
            title='Sentiment Trend Over Time',
            color_discrete_map={
                'Positive': '#48bb78',
                'Neutral': '#ecc94b',
                'Negative': '#f56565'
            }
        )
        fig.update_layout(
            autosize=True,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig.to_html(full_html=False)

    # =====================================================
    # FIX HISTOGRAM
    # =====================================================
    def rating_histogram(self):
        fig = px.histogram(
            self.df,
            x='rating',
            color='sentiment',
            nbins=5,
            title='Rating Distribution'
        )
        fig.update_layout(
            autosize=True,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig.to_html(full_html=False)

    # =====================================================
    # FIX SCATTER PLOT
    # =====================================================
    def rating_scatter(self):
        fig = px.scatter(
            self.df,
            x='rating',
            y='review_length',
            color='sentiment',
            title='Rating vs Review Length'
        )
        fig.update_layout(
            autosize=True,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig.to_html(full_html=False)

    # =====================================================
    # FIX KMEANS
    # =====================================================
    def kmeans_clustering(self):
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(self.X_vectors)
        svd = TruncatedSVD(n_components=2, random_state=42)
        reduced = svd.fit_transform(self.X_vectors)
        
        cluster_df = pd.DataFrame({
            'x': reduced[:,0],
            'y': reduced[:,1],
            'cluster': clusters.astype(str)
        })
        
        fig = px.scatter(
            cluster_df,
            x='x',
            y='y',
            color='cluster',
            title='K-Means Clustering'
        )
        fig.update_layout(
            autosize=True,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig.to_html(full_html=False)

    # =====================================================
    # STATIC IMAGES
    # =====================================================
    def generate_wordcloud(self, sentiment):
        text = ' '.join(self.df[self.df['sentiment'] == sentiment]['verified_reviews'].astype(str))
        if text.strip() == "":
            return
        
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10,5))
        plt.imshow(wc)
        plt.axis('off')
        plt.title(f'{sentiment} Word Cloud')
        
        save_path = os.path.join(self.save_dir, f'wordcloud_{sentiment.lower()}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"{sentiment} Word Cloud Saved")

    def confusion_matrix_chart(self):
        cm = confusion_matrix(self.df['feedback'], self.df['predicted_feedback'])
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print("Confusion Matrix Saved")

    def correlation_matrix_chart(self):
        corr = self.df[['rating', 'feedback', 'review_length']].corr()
        plt.figure(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        
        save_path = os.path.join(self.save_dir, 'correlation_matrix.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print("Correlation Matrix Saved")


if __name__ == "__main__":
    dashboard = DashboardVisualizer()
    print("\nGenerating Visualizations...\n")
    dashboard.sentiment_distribution()
    dashboard.sentiment_trend()
    dashboard.rating_histogram()
    dashboard.rating_scatter()
    dashboard.generate_wordcloud('Positive')
    dashboard.generate_wordcloud('Neutral')
    dashboard.generate_wordcloud('Negative')
    dashboard.confusion_matrix_chart()
    dashboard.correlation_matrix_chart()
    dashboard.kmeans_clustering()
    print("\nAll Visualizations Generated Successfully!\n")