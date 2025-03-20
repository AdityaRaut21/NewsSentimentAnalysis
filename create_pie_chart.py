import plotly.graph_objs as go


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
