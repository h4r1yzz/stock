import numpy as np
import datetime as dt
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from prophet import Prophet
import financedatabase as fd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import ta

st. set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Portfolio Analysis")

@st.cache_data
def load_data():
    ticker_list = pd.concat([fd.ETFs().select().reset_index()[['symbol', 'name']],
                             fd.Equities().select().reset_index()[['symbol', 'name']]])
    ticker_list = ticker_list[ticker_list.symbol.notna()]
    ticker_list['symbol_name'] = ticker_list.symbol + ' - ' + ticker_list.name

    return ticker_list
ticker_list = load_data()

# Add technical indicators
def add_technical_indicators(data):
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    return data

# Fetch stock data
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()  # Use datetime.now() correctly
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data

# Process stock data
def process_data(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert('US/Eastern')
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data


with st.sidebar:
    sel_ticker = st.multiselect("Portfolio Builder", placeholder="Search tickers", options=ticker_list.symbol_name)
    sel_ticker_list = ticker_list[ticker_list.symbol_name.isin(sel_ticker)].symbol

    cols = st.columns(4)
    for i, ticker in enumerate(sel_ticker_list):
        try:
            cols[i % 4].image(
                'https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.', ''),
                width=65
            )
        except:
            cols[i % 4].subheader(ticker)

    cols = st.columns(2)
    sel_dt1 = cols[0].date_input('Start Date', value=dt.datetime(2024, 1, 1), format='YYYY-MM-DD')
    sel_dt2 = cols[1].date_input('End Date', value=dt.datetime(2024, 8, 6), format='YYYY-MM-DD')


    if len(sel_ticker) != 0:
        yfdata = yf.download(list(sel_ticker_list), start=sel_dt1, end=sel_dt2)['Close'].reset_index().melt(id_vars = ['Date'], var_name= 'ticker', value_name='price')
        yfdata['price_start'] = yfdata.groupby('ticker').price.transform('first')
        yfdata['price_pct_daily'] = yfdata.groupby('ticker').price.pct_change()
        yfdata['price_pct'] = (yfdata.price - yfdata.price_start) / yfdata.price_start
        yfdata['Close'] = yfdata['price']
        yfdata = add_technical_indicators(yfdata)

        # Sidebar Header
    st.sidebar.header('Real-Time Stock Prices')

    # Use Portfolio Builder's selected tickers (sel_ticker) for Real-Time Prices
    if sel_ticker:
        selected_symbols = ticker_list[ticker_list['symbol_name'].isin(sel_ticker)]['symbol'].tolist()

        # Display Real-Time Stock Prices for Portfolio Builder Selections
        for symbol in selected_symbols:
            real_time_data = fetch_stock_data(symbol, '1d', '1m')
            if not real_time_data.empty:
                real_time_data = process_data(real_time_data)

                last_price = real_time_data['Close'].iloc[-1]
                open_price = real_time_data['Open'].iloc[0]

                last_price = float(last_price)
                open_price = float(open_price)

                change = last_price - open_price
                pct_change = (change / open_price) * 100

                st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")
    else:
        st.sidebar.info("Select tickers in the Portfolio Builder to view real-time prices.")


@st.cache_data
def fetch_stock_data_cached(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)



tab1, tab2 = st.tabs(["Portfolio", "Calculator"])

if len(sel_ticker) == 0:
    st.info("Select tickers to view plots")
else:
    st.empty()

    with tab1:
        st.subheader("All Stocks Overview")
        fig = px.line(yfdata, x='Date', y='price_pct', color='ticker', markers=True)
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.update_layout(xaxis_title=None, yaxis_title="Percentage Change")
        fig.update_yaxes(tickformat=',.0%')
        fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True, conifg={"responsive": True})

        # Individual stock overview
        st.subheader('individual stocks')
        cols = st.columns(3)
        for i, ticker in enumerate(sel_ticker_list):
            try:
                cols[i % 3].image('https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.', ''), width=65)
            except:
                cols[i % 3].subheader(ticker)

            cols2 = cols[i % 3].columns(3)
            cols2[0].metric(label='50-Day Average', value="N/A" if yfdata.empty else round(yfdata[yfdata.ticker == ticker].price.tail(50).mean(), 2))
            cols2[1].metric(label='1-Year Low', value="N/A" if yfdata.empty else round(yfdata[yfdata.ticker == ticker].price.tail(365).min(), 2))
            cols2[2].metric(label='1-Year High', value="N/A" if yfdata.empty else round(yfdata[yfdata.ticker == ticker].price.tail(365).max(), 2))

            # Plot individual stock with SMA and EMA
            fig = px.line(yfdata[yfdata.ticker == ticker], x='Date', y='price', markers=True)
            fig.add_trace(go.Scatter(x=yfdata[yfdata.ticker == ticker]['Date'], 
                                    y=yfdata[yfdata.ticker == ticker]['SMA_20'], 
                                    mode='lines', name=f'{ticker} SMA_20'))
            fig.add_trace(go.Scatter(x=yfdata[yfdata.ticker == ticker]['Date'], 
                                    y=yfdata[yfdata.ticker == ticker]['EMA_20'], 
                                    mode='lines', name=f'{ticker} EMA_20'))
            
            fig.update_layout(xaxis_title=None, yaxis_title=None)
            cols[i % 3].plotly_chart(fig, use_container_width=True, config={"responsive": True})



    with tab2:
        cols_tab2 = st.columns((0.2, 0.8))
        total_inv = 0
        amounts = {}

        for i, ticker in enumerate(sel_ticker_list):
            cols = cols_tab2[0].columns((0.1, 0.3))
            try:
                cols[0].image('https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.', ''), width=65)
            except:
                cols[0].subheader(ticker)

            amount = cols[1].number_input(f'Amount for {ticker}', key=ticker, step=50)
            total_inv += amount
            amounts[ticker] = amount

        cols_tab2[1].subheader(f'Total Investment: ${total_inv}')
        cols_goal = cols_tab2[1].columns((0.08, 0.2, 0.7))
        cols_goal[0].text('')
        cols_goal[0].subheader('Goal: ')
        goal = cols_goal[1].number_input('', key='goal', step=50)

        # Update DataFrame with amounts
        df = yfdata.copy()
        df['amount'] = df['ticker'].map(amounts) * (1 + df['price_pct'])

        # Calculate total investments per date
        dfsum = df.groupby('Date')['amount'].sum().reset_index()

        # Plotly Chart
        fig = px.area(df, x='Date', y='amount', color='ticker')
        fig.add_hline(y=goal, line_color='rgb(57,255,20)', line_dash='dash', line_width=3)

        if dfsum[dfsum['amount'] >= goal].shape[0] == 0:
            cols_tab2[1].warning("The goal can't be reached within this time frame. Either change the goal amount or the time frame.")
        else:
            goal_date = dfsum[dfsum['amount'] >= goal].iloc[0]['Date']
            fig.add_vline(x=goal_date, line_color='rgb(57,255,20)', line_dash='dash', line_width=3)
            fig.add_trace(go.Scatter(
                x=[goal_date + dt.timedelta(days=7)],
                y=[goal * 1.1],
                text=[goal_date.strftime('%Y-%m-%d')],
                mode='text',
                name="Goal",
                textfont=dict(color="rgb(57,255,20)", size=20)
            ))
            fig.update_layout(xaxis_title=None, yaxis_title=None)
            cols_tab2[1].plotly_chart(fig, use_container_width=True)