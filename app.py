import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import nltk
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime

# ✅ Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    with st.spinner("Downloading VADER sentiment lexicon..."):
        nltk.download('vader_lexicon')

# ✅ Cache model loading
@st.cache_resource
def load_lstm_model():
    return load_model(r'C:\Users\shubh\AppData\Local\Programs\Python\Stock market\Stock Prediction Model.keras')

model = load_lstm_model()

# ✅ Streamlit config
st.set_page_config(page_title="📈 Stock Market Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>📈 Stock Market Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# ✅ Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Market", 
    "📊 Prediction Dashboard", 
    "🔮 Forecast", 
    "📰 News & Sentiment",
    "💰 Company Financials"
])

# ✅ Sidebar
st.sidebar.title("Controls")
stock = st.sidebar.selectbox(
    'Choose stock symbol:',
    ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFIBEAM.NS', 'GOOG', 'AAPL', 'MSFT', 'TSLA', 'AMZN']
)
forecast_days = st.sidebar.slider('Select forecast period (days):', 7, 90, 30)

ticker_to_name = {
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'Tata Consultancy Services', 'INFY.NS': 'Infosys',
    'HDFCBANK.NS': 'HDFC Bank', 'ICICIBANK.NS': 'ICICI Bank', 'GOOG': 'Google', 'AAPL': 'Apple',
    'MSFT': 'Microsoft', 'TSLA': 'Tesla', 'AMZN': 'Amazon', 'INFIBEAM.NS': 'Infibeam Avenues'
}
company_name = ticker_to_name.get(stock, stock)

start = '2012-01-01'
end = datetime.today().strftime('%Y-%m-%d')

# 📈 Live Market Tab
with tab1:
    st.subheader("📊 Indian Market Indices (Live)")
    try:
        nifty_data = yf.Ticker("^NSEI").history(period="1d", interval="1m")
        sensex_data = yf.Ticker("^BSESN").history(period="1d", interval="1m")
        if not nifty_data.empty and not sensex_data.empty:
            st.metric("Nifty 50", f"{nifty_data['Close'].dropna().iloc[-1]:.2f}")
            st.metric("Sensex", f"{sensex_data['Close'].dropna().iloc[-1]:.2f}")
    except Exception as e:
        st.warning(f"Error fetching indices: {e}")

    st.subheader(f"📈 Live Market Stats for {stock}")
    try:
        live_data = yf.download(tickers=stock, period='1d', interval='1m')
        close_series = live_data['Close'].dropna() if 'Close' in live_data else pd.Series()
        high_series = live_data['High'].dropna() if 'High' in live_data else pd.Series()
        low_series = live_data['Low'].dropna() if 'Low' in live_data else pd.Series()

        if not close_series.empty and not high_series.empty and not low_series.empty:
            latest_close = float(close_series.iloc[-1])
            day_high = float(high_series.max())
            day_low = float(low_series.min())

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{latest_close:.2f}")
            col2.metric("Day High", f"₹{day_high:.2f}")
            col3.metric("Day Low", f"₹{day_low:.2f}")
        else:
            st.warning("⚠️ Live data not available or incomplete for selected stock.")
    except Exception as e:
        st.warning(f"Error fetching live stock data: {e}")

# 📊 Prediction Tab
with tab2:
    data = yf.download(stock, start=start, end=end)
    st.subheader(f"Raw Data for {stock}")
    st.write(data.tail())

    train_len = int(len(data) * 0.80)
    data_train = data.Close[:train_len]
    data_test = data.Close[train_len:]

    scaler = MinMaxScaler()
    scaler.fit(np.array(data_train).reshape(-1, 1))

    past_100 = data_train.tail(100)
    data_test_full = pd.concat([past_100, data_test])
    data_test_scaled = scaler.transform(np.array(data_test_full).reshape(-1, 1))

    st.subheader("📈 Moving Averages")
    fig = plt.figure(figsize=(10, 5))
    plt.plot(data.Close, label='Closing')
    plt.plot(data.Close.rolling(50).mean(), label='MA50')
    plt.plot(data.Close.rolling(100).mean(), label='MA100')
    plt.plot(data.Close.rolling(200).mean(), label='MA200')
    plt.legend()
    st.pyplot(fig)

    x, y = [], []
    for i in range(100, len(data_test_scaled)):
        x.append(data_test_scaled[i-100:i])
        y.append(data_test_scaled[i, 0])
    x, y = np.array(x), np.array(y)

    predicted = model.predict(x)
    predicted = scaler.inverse_transform(predicted)
    y_actual = scaler.inverse_transform(y.reshape(-1, 1))

    st.subheader("📊 Prediction Accuracy")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_actual, predicted)):.2f}")
    st.metric("MAE", f"{mean_absolute_error(y_actual, predicted):.2f}")

    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(y_actual, 'r', label='Original')
    plt.plot(predicted, 'b', label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig3)

# 🔮 Forecast Tab
with tab3:
    st.subheader(f"🔮 Next {forecast_days} Days Price Forecast")
    required_input_len = 100

    if len(data_test_scaled) < required_input_len:
        st.warning("Not enough data for forecast. Padding data for forecast...")
        last_100 = np.pad(
            data_test_scaled.flatten(), (required_input_len - len(data_test_scaled), 0), mode='edge'
        ).reshape(-1, 1)
    else:
        last_100 = data_test_scaled[-required_input_len:]

    future_input = list(last_100.reshape(-1))
    future_preds = []

    try:
        for _ in range(forecast_days):
            curr_input = np.array(future_input[-100:]).reshape(1, 100, 1)
            pred = model.predict(curr_input, verbose=0)[0][0]
            future_preds.append(pred)
            future_input.append(pred)

        if future_preds:
            future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

            fig4 = plt.figure(figsize=(10, 5))
            plt.plot(future_dates, future_preds, marker='o')
            plt.title(f"Next {forecast_days} Days Forecast")
            plt.xlabel("Date")
            plt.ylabel("Predicted Price")
            plt.xticks(rotation=45)
            st.pyplot(fig4)
        else:
            st.error("Prediction failed: no output generated.")
    except Exception as e:
        st.error(f"⚠️ Forecasting error: {e}")

# 📰 News & Sentiment Tab
with tab4:
    st.subheader(f"📰 Latest News and Sentiment for {company_name}")
    api_key = os.getenv("NEWSAPI_KEY", "dfb3d65cad714e439b93a05bee881f63")
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    pos, neu, neg = 0, 0, 0

    try:
        res = requests.get(url)
        news = res.json()
        if news.get("status") == "ok":
            sid = SentimentIntensityAnalyzer()
            for article in news["articles"]:
                title = article.get("title", "")
                link = article.get("url", "#")
                sentiment = sid.polarity_scores(title)
                compound = sentiment["compound"]
                if compound > 0.05:
                    label, pos = "🟢 Positive", pos + 1
                elif compound < -0.05:
                    label, neg = "🔴 Negative", neg + 1
                else:
                    label, neu = "⚪ Neutral", neu + 1
                st.markdown(f"**{label}**: [{title}]({link})")

            fig_pie, ax = plt.subplots()
            ax.pie([pos, neu, neg], labels=["Positive", "Neutral", "Negative"],
                   colors=["#00cc44", "#999999", "#ff4d4d"], autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig_pie)
        else:
            st.error("⚠️ Failed to fetch news. Check API key.")
    except Exception as e:
        st.error(f"⚠️ Error fetching news: {e}")

# 💰 Company Financials Tab
with tab5:
    st.subheader(f"💰 Revenue vs Net Income of {company_name}")
    try:
        fin_data = yf.Ticker(stock)
        income_stmt = fin_data.financials
        if not income_stmt.empty and 'Total Revenue' in income_stmt.index and 'Net Income' in income_stmt.index:
            revenue = income_stmt.loc['Total Revenue']
            net_income = income_stmt.loc['Net Income']

            # Reverse to chronological order (oldest to latest)
            revenue = revenue.sort_index()
            net_income = net_income.sort_index()
            years = revenue.index.strftime('%Y')

            # Combine into a single DataFrame
            fin_df = pd.DataFrame({
                'Year': years,
                'Revenue (₹B)': revenue.values / 1e9,
                'Net Income (₹B)': net_income.values / 1e9
            })

            # Plot the data
            fig_fin, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(fin_df['Year'], fin_df['Revenue (₹B)'], 'bo-', label='Revenue')
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Revenue (₹ Billion)", color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            ax2 = ax1.twinx()
            ax2.plot(fin_df['Year'], fin_df['Net Income (₹B)'], 'or--', label='Net Income')
            ax2.set_ylabel("Net Income (₹ Billion)", color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            plt.title("Revenue vs Net Income (Chronological Order)")
            fig_fin.tight_layout()
            st.pyplot(fig_fin)

            # 📥 Download Button
            csv = fin_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Financial Data as CSV",
                data=csv,
                file_name=f'{company_name}_financials.csv',
                mime='text/csv'
            )
        else:
            st.warning("Financial data not available or incomplete for this company.")
    except Exception as e:
        st.error(f"⚠️ Error fetching financials: {e}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Keras")
