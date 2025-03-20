# sentiment_analysis.py

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline  # Import pipeline for multilingual analysis

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Initialize the multilingual sentiment analysis model
multilingual_sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(text, language):
    sia = SentimentIntensityAnalyzer()
    
    if language == 'en':
        sentiment_scores = sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        if compound_score >= 0.05:
            return "Positive 😊", compound_score
        elif compound_score <= -0.05:
            return "Negative ☹️", compound_score
        else:
            return "Neutral 😐", compound_score
    elif language == 'hi':
        # Use the multilingual sentiment analyzer for Hindi
        result = multilingual_sentiment_analyzer(text)
        label = result[0]['label']
        score = result[0]['score']
        
        if "positive" in label.lower():
            return "सकारात्मक 😊", score  # Positive
        elif "negative" in label.lower():
            return "नकारात्मक ☹️", score  # Negative
        else:
            return "तटस्थ 😐", score  # Neutral
