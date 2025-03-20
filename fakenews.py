import os
from datetime import datetime, timedelta

import torch
import dash_bootstrap_components as dbc
import feedparser  # For RSS Feeds
import joblib
import nltk
import plotly.graph_objs as go
import requests
from dash import Dash, Input, Output, dcc, html
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Load trained Fake News Detection model
fake_news_model = joblib.load("fake_news_xgboost.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load multilingual sentiment analysis model
multilingual_sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# RSS Feed URLs (English & Hindi sources)
RSS_FEEDS = {
    "finance_en": "https://rss.cnn.com/rss/money_news_international.rss",
    "finance_hi": "https://www.bbc.com/hindi/index.xml",
    "healthcare_en": "https://www.who.int/rss-feeds/news-english.xml",
    "healthcare_hi": "https://www.bbc.com/hindi/science-and-environment/index.xml",
    "education_en": "https://www.theguardian.com/education/rss",
    "education_hi": "https://www.bbc.com/hindi/india/index.xml"
}

# ✅ Fetch live news using RSS Feeds
def fetch_rss_news(category, language):
    feed_key = f"{category}_{language}"
    feed_url = RSS_FEEDS.get(feed_key, None)
    if not feed_url:
        return []

    feed = feedparser.parse(feed_url)
    articles = []

    for entry in feed.entries[:10]:  # Fetch top 10 news
        description = entry.summary if 'summary' in entry else "No description available"
        short_description = (description[:200] + "...") if len(description) > 200 else description  # ✅ Truncate Description

        articles.append({
            'title': entry.title,
            'description': short_description,
            'url': entry.link,
            'published_at': entry.published if hasattr(entry, 'published') else "Unknown",
            'language': language
        })

    return articles

# ✅ Fetch live news using NewsAPI (if RSS fails)
def fetch_news_api(api_key, category, language):
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    base_url = "https://newsapi.org/v2/everything"

    params = {
        'apiKey': api_key,
        'q': category,
        'language': language,
        'from': one_week_ago,
        'sortBy': 'publishedAt'
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        return [
            {
                'title': article['title'],
                'description': (article['description'][:200] + "...") if article['description'] and len(article['description']) > 200 else article['description'] or "No description available",
                'url': article['url'],
                'published_at': article['publishedAt'],
                'language': language
            } for article in data['articles'][:10]
        ]
    else:
        print(f"Error fetching NewsAPI: {response.status_code}")
        return []

# ✅ Fake News Detection
def detect_fake_news(text):
    text_transformed = tfidf_vectorizer.transform([text])
    prediction = fake_news_model.predict(text_transformed)[0]
    return "Fake News ❌" if prediction == 1 else "Real News ✅"

# ✅ Perform sentiment analysis (English & Hindi)
def analyze_sentiment(text, language):
    if language == 'en':
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            return "Positive 😊", compound_score
        elif compound_score <= -0.05:
            return "Negative ☹️", compound_score
        else:
            return "Neutral 😐", compound_score
    elif language == 'hi':
        result = multilingual_sentiment_analyzer(text)
        label = result[0]['label']
        score = result[0]['score']

        if "positive" in label.lower():
            return "सकारात्मक 😊", score  # Positive
        elif "negative" in label.lower():
            return "नकारात्मक ☹️", score  # Negative
        else:
            return "तटस्थ 😐", score  # Neutral

# ✅ Collect sentiment data
def collect_sentiment_data(api_key, category):
    sentiment_data = []

    for language in ['en', 'hi']:
        rss_news = fetch_rss_news(category, language)
        api_news = fetch_news_api(api_key, category, language)

        news_articles = rss_news if rss_news else api_news

        for data in news_articles:
            title = data['title']
            title_sentiment, title_score = analyze_sentiment(title, language)
            fake_news_label = detect_fake_news(title)

            data.update({
                'title_sentiment': title_sentiment,
                'title_score': title_score,
                'fake_news': fake_news_label
            })

            sentiment_data.append(data)

    return sentiment_data

# ✅ Generate pie chart for sentiment distribution
def create_pie_chart(sentiment_data, category):
    sentiment_count = {'Positive 😊': 0, 'Negative ☹️': 0, 'Neutral 😐': 0}
    sentiment_count_hi = {'सकारात्मक 😊': 0, 'नकारात्मक ☹️': 0, 'तटस्थ 😐': 0}

    for data in sentiment_data:
        if data['language'] == 'en':
            sentiment_count[data['title_sentiment']] += 1
        elif data['language'] == 'hi':
            sentiment_count_hi[data['title_sentiment']] += 1

    sentiment_count.update(sentiment_count_hi)

    pie_chart = go.Figure(data=[go.Pie(labels=list(sentiment_count.keys()), values=list(sentiment_count.values()),
                                       title=f'Sentiment Distribution for {category.capitalize()}')])
    return pie_chart

# ✅ Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ✅ Layout
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Live News Sentiment Analysis with Fake News Detection"), className="text-center")]),

    dbc.Row([dbc.Col([
        html.Label("Select News Category:"),
        dcc.Dropdown(id='category-dropdown',
                     options=[
                         {'label': 'Finance', 'value': 'finance'},
                         {'label': 'Healthcare', 'value': 'healthcare'},
                         {'label': 'Education', 'value': 'education'}
                     ],
                     value='finance',
                     multi=False,
                     clearable=False,
                     className="mb-4"),
    ], width=6)]),

    dbc.Row([dbc.Col([
        html.Button('Fetch Latest News', id='fetch-news-button', n_clicks=0, className="btn btn-primary")
    ], width=2)]),

    dbc.Row([dbc.Col([dcc.Graph(id='sentiment-pie-chart')])]),
    dbc.Row([dbc.Col(html.Div(id='news-output'))]),
])

# ✅ Callback to update UI
@app.callback(
    Output('sentiment-pie-chart', 'figure'),
    Output('news-output', 'children'),
    Input('fetch-news-button', 'n_clicks'),
    Input('category-dropdown', 'value')
)
def update_pie_chart(n_clicks, category):
    api_key = "033302b4ad3c4ca1bc664e1c784bb622"  
    sentiment_data = collect_sentiment_data(api_key, category)
    pie_chart = create_pie_chart(sentiment_data, category)

    news_output = [
        html.Div([
            html.H4(data['title']),
            html.P(f"Description: {data['description']}"),
            html.P(f"Sentiment: {data['title_sentiment']} (Confidence: {data['title_score']:.2f})"),
            html.P(f"Fake News Detection: {data['fake_news']}"),
            html.P(f"URL: {data['url']}"),
            html.P(f"Published at: {data['published_at']}"),
            html.Hr()
        ]) for data in sentiment_data
    ]

    return pie_chart, news_output

# ✅ Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
