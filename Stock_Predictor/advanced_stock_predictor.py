import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import yfinance as yf
from flask import Flask, request, render_template_string, jsonify
from datetime import datetime, timedelta
import ta
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import requests

app = Flask(__name__)

STOCKS = {
    'AMZN': 'Amazon',
    'AAPL': 'Apple',
    'NVDA': 'NVIDIA',
    'ETC-USD': 'Ethereum Classic'
}

def fetch_stock_data(symbol, period='1y'):
    df = yf.download(symbol, period=period)
    return df

def add_technical_indicators(df):
    df['SMA'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['BB_up'], df['BB_mid'], df['BB_low'] = ta.volatility.bollinger_hband_indicator(df['Close']), ta.volatility.bollinger_mavg(df['Close']), ta.volatility.bollinger_lband_indicator(df['Close'])
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
    return df

def prepare_data(data, time_steps=60):
    feature_columns = ['Close', 'Volume', 'SMA', 'RSI', 'MACD', 'BB_up', 'BB_low', 'ADX']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature_columns])
    
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 0])  # predicting the 'Close' price
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler, feature_columns

def create_model(X):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_stock_chart(df, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=(title, 'Volume'), 
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'),
                  row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

    fig.update_layout(height=600, width=1000, 
                      template='plotly_dark',
                      margin=dict(l=50, r=50, t=50, b=50),
                      xaxis_rangeslider_visible=False)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return pio.to_html(fig, full_html=False, config={'responsive': True})

def create_prediction_chart(historical_data, predicted_prices, symbol):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Historical Price'))

    future_dates = pd.date_range(start=historical_data.index[-1] + timedelta(days=1), periods=len(predicted_prices))
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines+markers', name='Predicted Price'))

    fig.update_layout(
        title=f'{symbol} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        height=400,  # Reduced height
        # width=100%,  # Remove fixed width
        template='plotly_white',  # Changed to a lighter theme
        margin=dict(l=40, r=40, t=40, b=40),  # Adjusted margins
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)  # Horizontal legend at the top
    )

    # Make the chart responsive
    fig.update_layout(autosize=True)

    return pio.to_html(fig, full_html=False, config={'responsive': True, 'displayModeBar': False})

def create_mini_chart(symbol, period='1mo'):
    data = yf.download(symbol, period=period)
    fig = go.Figure(data=[go.Scatter(x=data.index, y=data['Close'], mode='lines')])
    fig.update_layout(
        height=100, width=200,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return pio.to_html(fig, full_html=False, config={'displayModeBar': False})

def format_datetime(value, format='%Y-%m-%d %H:%M:%S'):
    if value is None:
        return ''
    if isinstance(value, int):
        value = datetime.fromtimestamp(value)
    return value.strftime(format)

app.jinja_env.filters['datetime'] = format_datetime

def fetch_stock_news(symbol):
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.now().date()}&to={datetime.now().date()}&token=cr8vve9r01qmiu0b0l0gcr8vve9r01qmiu0b0l10"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()[:5]  # Return top 5 news items
    return []

@app.route('/')
def index():
    stock_data = {}
    all_news = {}
    for symbol, name in STOCKS.items():
        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get('currentPrice', 'N/A')
        change = info.get('regularMarketChangePercent', 'N/A')
        
        price = float(price) if price != 'N/A' else 'N/A'
        change = float(change) if change != 'N/A' else 0
        
        stock_data[symbol] = {
            'name': name,
            'price': price,
            'change': change,
            'mini_chart': create_mini_chart(symbol)
        }
        
        all_news[symbol] = fetch_stock_news(symbol)
    
    return render_template_string(INDEX_TEMPLATE, stocks=STOCKS, stock_data=stock_data, all_news=all_news)

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    days_to_predict = int(request.form['days_to_predict'])
    
    # Fetch historical data (1 year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Add technical indicators
    data = add_technical_indicators(data)
    data = data.dropna()

    # Prepare data for LSTM
    X, y, scaler, feature_columns = prepare_data(data)

    # Create and train the model
    model = create_model(X)
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # Make predictions
    last_60_days = data[feature_columns].tail(60).values
    predicted_prices = []

    for _ in range(days_to_predict):
        scaled_data = scaler.transform(last_60_days[-60:])
        X_pred = np.array([scaled_data])
        predicted_scaled_price = model.predict(X_pred)
        predicted_price = scaler.inverse_transform(np.hstack([predicted_scaled_price, np.zeros((1, len(feature_columns)-1))]))[0][0]
        predicted_prices.append(predicted_price)

        # Update last_60_days for the next prediction
        new_row = np.zeros((1, len(feature_columns)))
        new_row[0, 0] = predicted_price  # Set the predicted Close price
        last_60_days = np.vstack([last_60_days, new_row])

    # Create prediction chart
    prediction_chart = create_prediction_chart(data, predicted_prices, symbol)

    return jsonify({
        'symbol': symbol,
        'last_price': data['Close'].iloc[-1],
        'predicted_prices': predicted_prices,
        'prediction_chart': prediction_chart
    })

# @app.route('/get_more_news')
# def get_more_news():
#     symbol = request.args.get('symbol')
#     news = fetch_stock_news(symbol)[1:]  # Skip the first item as it's already shown
#     return jsonify({'news': news})

INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YongStock - Advanced Stock Predictor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'primary': '#0A2342',    // Deep navy blue
                        'secondary': '#2CA6A4',  // Teal accent
                        'accent': '#77A6F7',     // Gold
                        'background': '#F0F0F0', // Light gray background
                        'text-primary': '#FFFFFF', // White text
                        'text-secondary': '#333333' // Dark gray text
                    }
                }
            }
        }
    </script>
    <style>
        #results {
            max-width: 100%;
            overflow-x: hidden;
        }
        #prediction-chart {
            width: 250%;
            max-width: 250%;
            overflow-x: auto;
        }
        .shadow-luxury {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .hover-luxury:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        
    </style>
</head>
<body class="bg-background text-text-secondary font-sans">
    <nav class="bg-primary text-text-primary shadow-lg">
        <div class="container mx-auto px-6 py-3 flex justify-between items-center">
            <a href="/" class="font-bold text-xl text-white">YongStock</a>
            <div class="flex items-center space-x-4">
                <a href="#" class="hover:text-accent transition">Dashboard</a>
                <a href="#" class="hover:text-accent transition">Markets</a>
                <a href="#" class="hover:text-accent transition">News</a>
                <a href="#" class="hover:text-accent transition">About</a>
            </div>
        </div>
    </nav>

    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-primary mb-8 border-b-2 border-accent pb-2">Advanced Stock Predictor</h1>

        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {% for symbol, data in stock_data.items() %}
            
            
            <div class="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition border-l-4 border-accent">
                <h3 class="font-semibold text-lg mb-2 text-primary">{{ data['name'] }} ({{ symbol }})</h3>
                <p class="text-3xl font-bold mb-2 text-secondary">
                    {% if data['price'] != 'N/A' %}
                        ${{ "%.2f"|format(data['price']) }}
                    {% else %}
                        N/A
                    {% endif %}
                </p>
                <p class="{{ 'text-green-500' if data['change'] > 0 else 'text-red-500' if data['change'] < 0 else 'text-gray-500' }} font-semibold">
                    {% if data['change'] != 'N/A' %}
                        {{ "%.2f"|format(data['change']) }}% 
                        {% if data['change'] > 0 %}▲{% elif data['change'] < 0 %}▼{% else %}─{% endif %}
                    {% else %}
                        N/A
                    {% endif %}
                </p>
                <div class="mt-4">{{ data['mini_chart'] | safe }}</div>
            </div>
            {% endfor %}
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            <div class="lg:col-span-2">
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <h2 class="text-2xl font-semibold text-primary mb-4">Stock Price Prediction</h2>
                    <form id="prediction-form" class="mb-4">
                        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                            <div>
                                <label for="symbol" class="block mb-2 font-medium">Stock:</label>
                                <select id="symbol" name="symbol" class="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-secondary">
                                    {% for symbol, name in stocks.items() %}
                                        <option value="{{ symbol }}">{{ name }}</option>
                                    {% endfor %}
                                </select>
                            </div>
                            <div>
                                <label for="days_to_predict" class="block mb-2 font-medium">Days to Predict:</label>
                                <input type="number" id="days_to_predict" name="days_to_predict" min="1" max="30" value="7" required class="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-secondary">
                            </div>
                            <div class="flex items-end">
                            <button type="submit" class="bg-accent text-primary px-6 py-2 rounded hover:bg-opacity-90 transition w-full font-semibold">
                                <i class="fas fa-chart-line mr-2"></i>Predict
                            </button>
                            </div>
                        </div>
                    </form>
                </div>
                <div id="results" class="hidden bg-white rounded-lg shadow-md p-6 border-t-4 border-accent">
                    <h2 class="text-2xl font-semibold text-primary mb-4">Prediction Results</h2>
                    <div id="prediction-chart" class="mb-6"></div>
                    <div id="results-content"></div>
                </div>
            </div>
            <div>
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-2xl font-semibold text-primary mb-4">Market News</h2>
                    {% for symbol, news_items in all_news.items() %}
                    <div class="mb-6">
                        <h3 class="text-lg font-semibold mb-2">{{ stocks[symbol] }} ({{ symbol }})</h3>
                        <ul class="space-y-4" id="news-list-{{ symbol }}">
                            {% if news_items %}
                            <li>
                                <a href="{{ news_items[0].url }}" target="_blank" class="block hover:bg-gray-100 p-2 rounded transition">
                                    <h4 class="font-semibold text-accent">{{ news_items[0].headline }}</h4>
                                    <p class="text-sm text-gray-600">{{ news_items[0].summary[:100] }}...</p>
                                    <p class="text-xs text-gray-400 mt-1">{{ news_items[0].datetime | datetime('%Y-%m-%d %H:%M') }}</p>
                                </a>
                            </li>
                            {% else %}
                            <li>No recent news available.</li>
                            {% endif %}
                        </ul>
                        {% if news_items|length > 1 %}
                        <button class="toggle-news mt-2 text-secondary hover:text-primary transition" data-symbol="{{ symbol }}">
                            Show More
                        </button>
                        <script>
                            var allNews{{ symbol }} = {{ news_items|tojson|safe }};
                        </script>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>

    <footer class="bg-primary text-text-primary py-8">
        <div class="container mx-auto px-6">
            <div class="flex flex-wrap justify-between">
                <div class="w-full md:w-1/4 mb-6 md:mb-0">
                    <h3 class="text-xl font-bold mb-2">YongStock</h3>
                    <p class="text-sm">Advanced stock prediction and market analysis platform.</p>
                </div>
                <div class="w-full md:w-1/4 mb-6 md:mb-0">
                    <h4 class="text-lg font-semibold mb-2">Quick Links</h4>
                    <ul class="text-sm">
                        <li><a href="#" class="hover:text-secondary transition">Home</a></li>
                        <li><a href="#" class="hover:text-secondary transition">About Us</a></li>
                        <li><a href="#" class="hover:text-secondary transition">Contact</a></li>
                        <li><a href="#" class="hover:text-secondary transition">Terms of Service</a></li>
                    </ul>
                </div>
                <div class="w-full md:w-1/4 mb-6 md:mb-0">
                    <h4 class="text-lg font-semibold mb-2">Follow Us</h4>
                    <div class="flex space-x-4">
                        <a href="#" class="hover:text-secondary transition"><i class="fab fa-twitter"></i></a>
                        <a href="#" class="hover:text-secondary transition"><i class="fab fa-facebook"></i></a>
                        <a href="#" class="hover:text-secondary transition"><i class="fab fa-linkedin"></i></a>
                    </div>
                </div>
            </div>
            <div class="mt-8 text-center text-sm">
                <p>&copy; 2024 YongStock. All rights reserved.</p>
            </div>
        </div>
    </footer>

    <script>
        $(document).ready(function() {
            console.log("Documnet ready"); // Debug log

            $('#prediction-form').on('submit', function(e) {
                e.preventDefault();
                var formData = $(this).serialize();
                
                $.ajax({
                    url: '/predict',
                    method: 'POST',
                    data: formData,
                    beforeSend: function() {
                        $('#results').addClass('hidden');
                        // Add a loading indicator here if desired
                    },
                    success: function(response) {
                        var resultsHtml = `
                            <p><strong>Stock:</strong> ${response.symbol}</p>
                            <p><strong>Last Known Price:</strong> $${response.last_price.toFixed(2)}</p>
                            <h3 class="font-semibold mt-4">Predicted Prices:</h3>
                            <ul>
                        `;
                        response.predicted_prices.forEach((price, index) => {
                            const change = ((price - response.last_price) / response.last_price * 100).toFixed(2);
                            const changeClass = change >= 0 ? 'text-green-500' : 'text-red-500';
                            const changeSymbol = change >= 0 ? '▲' : '▼';
                            resultsHtml += `
                                <li>Day ${index + 1}: $${price.toFixed(2)} 
                                    <span class="${changeClass}">${changeSymbol} ${Math.abs(change)}%</span>
                                </li>
                            `;
                        });
                        resultsHtml += '</ul>';
                        
                        $('#results-content').html(resultsHtml);
                        $('#prediction-chart').html(response.prediction_chart);
                        $('#results').removeClass('hidden');
                    },
                    error: function(xhr, status, error) {
                        alert('An error occurred: ' + error);
                    }
                });
            });

            // Toggle news functionality
            $('.toggle-news').on('click', function() {
                var symbol = $(this).data('symbol');
                var newsListId = '#news-list-' + symbol;
                var button = $(this);
                var allNews = window['allNews' + symbol];
                
                if (button.text() === 'Show More') {
                    var newsHtml = allNews.slice(1).map(function(item) {
                        return `
                            <li>
                                <a href="${item.url}" target="_blank" class="block hover:bg-gray-100 p-2 rounded transition">
                                    <h4 class="font-semibold text-accent">${item.headline}</h4>
                                    <p class="text-sm text-gray-600">${item.summary.slice(0, 100)}...</p>
                                    <p class="text-xs text-gray-400 mt-1">${new Date(item.datetime * 1000).toLocaleString()}</p>
                                </a>
                            </li>
                        `;
                    }).join('');

                    $(newsListId).append(newsHtml);
                    button.text('Show Less');
                } else {
                    $(newsListId).find('li:not(:first-child)').remove();
                    button.text('Show More');
                }
            });
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)