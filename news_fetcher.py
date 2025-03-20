import feedparser
import requests

from config import GNEWS_API_KEY, NEWSAPI_KEY, RSS_FEED_URLS


# Function to fetch news from GNews API
def fetch_gnews(query):
    url = f'https://gnews.io/api/v4/search?q={query}&token={GNEWS_API_KEY}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    # Normalize data
    normalized_articles = []
    for article in articles:
        normalized_articles.append({
            "title": article.get("title"),
            "description": article.get("description"),
            "url": article.get("url"),
            "published_at": article.get("publishedAt"),
            "source": "GNews",
            "language": "en"
        })
    return normalized_articles

# Function to fetch news from RSS feeds
def fetch_rss_feed(feed_url):
    feed = feedparser.parse(feed_url)
    articles = []
    
    # Get the feed title (this will be used as the source)
    feed_title = feed.feed.get("title", "Unknown Source")
    
    for entry in feed.entries:
        articles.append({
            "title": entry.title,
            "description": entry.summary,
            "url": entry.link,
            "published_at": entry.published,
            "source": feed_title,  # Use the feed's title for the source
            "language": "en"  # Set this according to your RSS feed
        })
    return articles

# General function to fetch articles
def fetch_articles(query):
    articles = []
    
    # Fetch from GNews
    articles.extend(fetch_gnews(query))
    
    # Fetch from RSS feeds
    for feed_url in RSS_FEED_URLS:
        articles.extend(fetch_rss_feed(feed_url))
    
    return articles
