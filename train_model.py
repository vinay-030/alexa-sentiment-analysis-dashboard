import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

def get_robust_stopwords():
    """
    Returns a customized list of English stopwords that removes important 
    sentiment-bearing words so they are NOT stripped during preprocessing.
    """
    all_stopwords = stopwords.words('english')
    # Words we MUST keep to correctly identify negative sentiment
    important_words_to_keep = [
        'not', 'no', 'bad', 'worst', 'terrible', 'horrible', 'awful', 'hate', 
        'isn', "isn't", 'aren', "aren't", 'wasn', "wasn't", 'weren', "weren't",
        'hasn', "hasn't", 'haven', "haven't", 'hadn', "hadn't", 'won', "won't",
        'wouldn', "wouldn't", 'don', "don't", 'doesn', "doesn't", 'didn', "didn't",
        'can', "can't", 'couldn', "couldn't", 'shouldn', "shouldn't", 'mightn', "mightn't",
        'mustn', "mustn't", 'against', 'but', 'nor'
    ]
    
    robust_stopwords = [word for word in all_stopwords if word not in important_words_to_keep]
    return set(robust_stopwords)

ps = PorterStemmer()
robust_stopwords_set = get_robust_stopwords()

def preprocess_text(text):
    """
    Robust NLP preprocessing:
    - Lowercase conversion
    - Punctuation removal
    - Extra space removal
    - Tokenization & Stopword removal (avoiding aggressive removal)
    - Stemming
    """
    text = str(text)
    # Remove punctuation but keep spaces and letters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove extra spaces and tokenize
    words = text.split()
    # Remove stopwords (using our robust list) and apply stemming
    processed_words = [ps.stem(word) for word in words if word not in robust_stopwords_set]
    return ' '.join(processed_words)

def main():
    print("==================================================")
    print("LOADING AND PREPROCESSING DATA")
    print("==================================================")
    
    # 1. Load dataset (handle nulls if any)
    df = pd.read_csv('amazon_alexa.tsv', sep='\t')
    df = df.dropna(subset=['verified_reviews', 'feedback'])

    # ==================================================
    # ADD SYNTHETIC NEGATIVE REVIEWS
    # ==================================================
    extra_negative_reviews = pd.DataFrame({
        'verified_reviews': [
            'worst product', 'bad product', 'terrible experience', 'very bad',
            'poor quality', 'hate this alexa', 'awful product', 'not good',
            'waste of money', 'horrible experience', 'worst device ever',
            'very disappointing', 'extremely bad', 'pathetic product',
            'bad sound quality', 'worst purchase', 'terrible alexa',
            'do not buy', 'useless device', 'bad experience'
        ],
        'feedback': [0] * 20
    })

    # Merge with original dataset
    df = pd.concat([df, extra_negative_reviews], ignore_index=True)
    print("\nSynthetic negative reviews added successfully.")
    
    # 2. Apply preprocessing
    print("Applying NLP preprocessing to reviews... (this may take a moment)")
    df['processed_review'] = df['verified_reviews'].apply(preprocess_text)
    
    # 3. Train-Test Split
    X = df['processed_review']
    y = df['feedback']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # 4. Feature Engineering
    print("\n==================================================")
    print("FEATURE ENGINEERING (TfidfVectorizer)")
    print("==================================================")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 5. Model Training (Fixing Class Imbalance)
    print("\n==================================================")
    print("MODEL TRAINING & EVALUATION")
    print("==================================================")
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
        "LinearSVC": LinearSVC(class_weight='balanced', random_state=42)
    }
    
    best_model_name = ""
    best_model = None
    best_f1_macro = 0
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
        report = classification_report(y_test, y_pred)
        print("Classification Report:\n", report)
        
        from sklearn.metrics import f1_score
        f1_mac = f1_score(y_test, y_pred, average='macro')
        if f1_mac > best_f1_macro:
            best_f1_macro = f1_mac
            best_model_name = name
            best_model = model
            
    print("\n==================================================")
    print("SAVING PIPELINE")
    print("==================================================")
    print(f"Best model selected: {best_model_name}")
    
    # Ensure it predicts negative words correctly
    test_words = ["worst", "terrible", "bad", "excellent", "good"]
    print("\nSanity Check Predictions:")
    for w in test_words:
        w_proc = preprocess_text(w)
        w_vec = tfidf.transform([w_proc])
        pred = best_model.predict(w_vec)[0]
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f" '{w}' -> {sentiment}")
    
    # Save best model and vectorizer
    pickle.dump(best_model, open('robust_sentiment_model.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
    print("\nSaved 'robust_sentiment_model.pkl' and 'tfidf_vectorizer.pkl'")

if __name__ == "__main__":
    main()
