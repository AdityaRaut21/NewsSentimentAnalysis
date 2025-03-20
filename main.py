from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import nltk
import plotly.graph_objs as go
import requests
from dash import Dash, Input, Output, dcc, html
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Load multilingual sentiment analysis model
multilingual_sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Fetch live news using NewsAPI
def get_news(api_key, category, language):
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
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# Perform sentiment analysis on a text and return sentiment with emojis
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

# Collect sentiment data for live news
def collect_sentiment_data(api_key, category):
    sentiment_data = []
    
    # Fetch news in both English and Hindi
    for language in ['en', 'hi']:
        news = get_news(api_key, category, language)
        if news and news['articles']:
            for article in news['articles'][:10]:  # Limit to 10 articles
                title = article['title']
                title_sentiment, title_score = analyze_sentiment(title, language)
                description = article['description'] or "No description available"
                url = article['url']
                published_at = article['publishedAt']
                
                sentiment_data.append({
                    'title': title,
                    'description': description,
                    'title_sentiment': title_sentiment,
                    'title_score': title_score,
                    'url': url,
                    'published_at': published_at,
                    'language': language
                })
    return sentiment_data

# Generate pie chart for sentiment distribution
def create_pie_chart(sentiment_data, category):
    sentiment_count = {'Positive 😊': 0, 'Negative ☹️': 0, 'Neutral 😐': 0}
    sentiment_count_hi = {'सकारात्मक 😊': 0, 'नकारात्मक ☹️': 0, 'तटस्थ 😐': 0}  # Hindi sentiments
    
    for data in sentiment_data:
        if data['language'] == 'en':
            sentiment_count[data['title_sentiment']] += 1
        elif data['language'] == 'hi':
            sentiment_count_hi[data['title_sentiment']] += 1
    
    # Combine English and Hindi sentiment counts
    sentiment_count.update(sentiment_count_hi)
    
    pie_chart = go.Figure(data=[go.Pie(labels=list(sentiment_count.keys()), values=list(sentiment_count.values()), 
                                       title=f'Sentiment Distribution for {category.capitalize()}')])
    return pie_chart

# Dash App for the Generalized Website
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout for the Dash app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Live News Sentiment Analysis"), className="text-center")
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select News Category:"),
            dcc.Dropdown(id='category-dropdown',
                         options=[
                             {'label': 'Finance', 'value': 'finance'},
                             {'label': 'Healthcare', 'value': 'healthcare'},
                             {'label': 'Education', 'value': 'education'}
                         ],
                         value='finance',  # Default value
                         multi=False,
                         clearable=False,
                         className="mb-4"),
        ], width=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Button('Fetch Latest News', id='fetch-news-button', n_clicks=0, className="btn btn-primary")
        ], width=2),
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='sentiment-pie-chart')
        ])
    ]),
    
    dbc.Row([
        dbc.Col(html.Div(id='news-output'))
    ]),
])

# Callback to update the pie chart and news output based on the selected category and live news
@app.callback(
    Output('sentiment-pie-chart', 'figure'),
    Output('news-output', 'children'),
    Input('fetch-news-button', 'n_clicks'),
    Input('category-dropdown', 'value')
)
def update_pie_chart(n_clicks, category):
    api_key = "033302b4ad3c4ca1bc664e1c784bb622"  # Your NewsAPI key
    sentiment_data = collect_sentiment_data(api_key, category)  # Fetch live news in both English and Hindi
    pie_chart = create_pie_chart(sentiment_data, category)
    
    # Prepare news output
    news_output = []
    for data in sentiment_data:
        if data['language'] == 'en':
            news_output.append(html.Div([
                html.H4(data['title']),
                html.P(f"Description: {data['description']}"),
                html.P(f"Sentiment: {data['title_sentiment']} (Confidence: {data['title_score']:.2f})"),
                html.P(f"URL: {data['url']}"),
                html.P(f"Published at: {data['published_at']}"),
                html.Hr()
            ]))
        elif data['language'] == 'hi':
            news_output.append(html.Div([
                html.H4(data['title']),
                html.P(f"विवरण: {data['description']}"),
                html.P(f"भावना: {data['title_sentiment']} (विश्वास: {data['title_score']:.2f})"),
                html.P(f"यूआरएल: {data['url']}"),
                html.P(f"प्रकाशित तिथि: {data['published_at']}"),
                html.Hr()
            ]))
    
    return pie_chart, news_output

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
