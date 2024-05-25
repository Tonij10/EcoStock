# This file represent the main source of EcoStock package. It contains a collection of functions for fetching, analyzing and visualizing economic and stock data.

# Import the necessaries libraries
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.subplots as sp
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings

# This function is used to fetch stock data from Yahoo Finance.
def get_stock_data(ticker, start_date, end_date):
    """
    This function fetches historical market data for a specific stock, over a specific time period, from Yahoo Finance.
    
    Parameters:
    ticker (str): The ticker symbol of the stock for which data is to be fetched.
    start_date (str): The start date of the time period for which data is to be fetched. The date must be in 'YYYY-MM-DD' format.
    end_date (str): The end date of the time period for which data is to be fetched. The date must be in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: A DataFrame containing the historical market data for the specified stock over the specified time period. 
                      The DataFrame includes data like open price, close price, high price, low price, volume, etc. for each trading day within the time period.

    """
    # The yf.download function from the yfinance library is used to fetch the data. 
    # The function takes the ticker symbol and the start and end dates as arguments and returns a DataFrame with the stock data.
    return yf.download(ticker, start=start_date, end=end_date)

# This function is used to plot a candlestick chart for a specific stock over a specific time period.
def plot_candlestick(ticker, start_date, end_date):
    """
    This function fetches historical market data for a specific stock, over a specific time period, and plots it as a candlestick chart.
    
    Parameters:
    ticker (str): The ticker symbol of the stock for which data is to be fetched and plotted.
    start_date (str): The start date of the time period for which data is to be fetched and plotted. The date must be in 'YYYY-MM-DD' format.
    end_date (str): The end date of the time period for which data is to be fetched and plotted. The date must be in 'YYYY-MM-DD' format.
    """
    # Fetch the stock data using the get_stock_data function
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Create a candlestick chart using the plotly library. The x-axis represents the date, and the y-axis represents the stock price.
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'])])

    # Add volume bars to the chart. The color of the bars is determined by whether the closing price is higher or lower than the opening price.
    fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], marker=dict(color=np.where(stock_data['Close'] >= stock_data['Open'], 'green', 'red')), yaxis='y2'))

    # Configure the layout of the chart. This includes the title, y-axis title, range slider and other features.
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Stock Price (USD)',
        shapes = [dict(x0='2022-01-01', x1='2022-12-31', y0=0, y1=1, xref='x', yref='paper', line_width=2)],
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis2=dict(domain=[0, 0.2], anchor='x', title='Volume')
    )

    # Display the plot
    return fig.show()

# This function is used to fetch stock data, calculate a 1-year moving average, and resample the data to annual frequency.
def moving_avg_stock_data(ticker, start_date, end_date):
    """
    This function fetches historical market data for a specific stock, calculates a 1-year moving average of the data and then resamples the data to annual frequency.
    
    Parameters:
    ticker (str): The ticker symbol of the stock for which data is to be fetched.
    start_date (str): The start year for which data is to be fetched. The year must be in 'YYYY' format.
    end_date (str): The end year for which data is to be fetched. The year must be in 'YYYY' format.

    Returns:
    pandas.DataFrame: A DataFrame containing the 1-year moving average stock data, resampled to annual frequency. 
                      The DataFrame includes data like open price, close price, high price, low price, volume, etc. for each year within the time period.

    """
    # Format the years as 'YYYY-MM-DD' to represent the start and end dates of each year
    start_date = f'{start_date}-01-01'
    end_date = f'{end_date}-12-31'

    # Fetch the daily stock data using the yf.download function from the yfinance library
    df = yf.download(ticker, start=start_date, end=end_date)

    # Calculate a 1-year moving average of the stock data. The window parameter is set to 252, which is the typical number of trading days in a year.
    df = df.rolling(window=252).mean()

    # Resample the data to annual frequency, taking the last observation of each year. The 'Y' parameter represents annual frequency.
    df = df.resample('YE').last()

    # Modify the index to only include the year, by extracting the year from the index dates
    df.index = df.index.year

    return df

# This function is used to fetch economic data from the World Bank.
def get_world_bank_data(indicator, country, start_date, end_date):
    """
    This function fetches economic data for a specific indicator and country from the World Bank, for a specific time period.
    
    Parameters:
    indicator (str): The indicator of interest. This is a unique ID assigned to each economic indicator in the World Bank database.
    country (str): The country ISO code. This is a unique ID assigned to each country in the World Bank database.
    start_date (str): The start date for the data. The date should be in 'YYYY' format.
    end_date (str): The end date for the data. The date should be in 'YYYY' format.

    Returns:
    pandas.DataFrame: A DataFrame containing the economic data. The DataFrame includes data like the indicator value for each year within the time period.

    """
    # Check if the indicator is valid
    if not indicator:
        print(f"Error: Unknown indicator input '{indicator}'")
        return pd.DataFrame()
    
    # Define the API URL for the World Bank data
    url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start_date}:{end_date}&format=json"

    # Send the HTTP request to the API URL and get the response
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the data from the response
        data = response.json()

        # Check if the data has at least two elements (the first element is metadata and the second element is the actual data)
        if len(data) < 2:
            print("Error: No data available for the given parameters")
            return pd.DataFrame()

        # Convert the data to a DataFrame and set the date as the index
        df = pd.DataFrame(data[1])
        df['date'] = pd.to_datetime(df['date']).dt.year
        df.set_index('date', inplace=True)

        # Extract the name of the indicator and the country from the nested dictionaries in the 'indicator' and 'country' columns
        df['indicator'] = df['indicator'].apply(lambda x: x['value'])
        df['country'] = df['country'].apply(lambda x: x['value'])

        return df
    else:
        print(f"Error: HTTP request failed with status code {response.status_code}")
        return pd.DataFrame()

# This function is used to fetch stock data from Yahoo Finance, calculate a 1-year moving average, fetch economic data from the World Bank, and plot the correlation between stock and economic data over time.
def plot_correlation(ticker, indicator, country, start_date, end_date):
    """
    This function fetches stock data for a specific ticker from Yahoo Finance and calculates a 1-year moving average. It also fetches economic data for a specific indicator and country from the World Bank. It then plots the correlation between the stock and economic data over time.
    
    Parameters:
    ticker (str): The ticker symbol of the stock.
    indicator (str): The indicator of interest for the World Bank data. This is a unique ID assigned to each economic indicator in the World Bank database.
    country (str): The country ISO code. This is a unique ID assigned to each country in the World Bank database.
    start_date (str): The start date for the data. The date should be in 'YYYY' format.
    end_date (str): The end date for the data. The date should be in 'YYYY' format.

    Returns:
    matplotlib.pyplot: A plot showing the correlation between the stock and economic data over time.
    """
    # Fetch the 1-year moving average stock data
    stock_data = moving_avg_stock_data(ticker, start_date, end_date)

    # Fetch the economic data from the World Bank
    econ_data = get_world_bank_data(indicator, country, start_date, end_date)
    
    # Get the econ_label from the econ_data DataFrame
    econ_label = econ_data['indicator'].iloc[0]

    # Determine the econ_format based on the econ_label
    if '$' in econ_label:
        econ_format = 'US$'
    elif '%' in econ_label:
        econ_format = '%'
    else:
        econ_format = None

    # Merge the stock data and the economic data
    merged_data = pd.merge(stock_data, econ_data, how='inner', left_index=True, right_index=True)

    # Check if merged_data is empty
    if merged_data.empty:
        print("Error: No matching data found for stock and economic data")
        return None

    # Calculate the correlation between the stock data and the economic data
    correlation = merged_data['Close'].corr(merged_data['value'])

    # Create the plot
    sns.set(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Increase the size of the plot

    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{ticker} Price', color=color)
    ax1.plot(stock_data.index, stock_data['Close'], color=color, label=f'{ticker} Price')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(econ_label, color=color)
    ax2.plot(econ_data.index, econ_data['value'], color=color, label=econ_label)
    ax2.tick_params(axis='y', labelcolor=color)

    # Format the y-tick labels based on econ_format
    if econ_format == '%':
        formatter = mticker.FuncFormatter(lambda x, pos: '{:.1f}%'.format(x))
    elif econ_format == 'US$':
        def human_format(num):
            magnitude = 0
            while abs(num) >= 1000:
                magnitude += 1
                num /= 1000.0
            return '${:.1f}{}'.format(num, ['', 'K', 'M', 'B', 'T'][magnitude])

        formatter = mticker.FuncFormatter(lambda x, pos: human_format(x))
    else:
        formatter = None

    if formatter is not None:
        ax2.yaxis.set_major_formatter(formatter)

    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
    plt.title(f'Correlation between {ticker} Price and {econ_label}: {correlation:.2f}')
    fig.tight_layout()
    
    return plt.show()

# This function is used to plot Bollinger Bands for a specific stock over a specified time period.
def plot_bollinger_bands(ticker, start_date, end_date):
    """
    This function fetches stock data for a specific ticker over a specified time period, calculates Bollinger Bands and plots them along with the stock price data.
    
    Parameters:
    ticker (str): The ticker symbol of the stock.
    start_date (str): The start date for the data. The date should be in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data. The date should be in 'YYYY-MM-DD' format.

    Returns:
    plotly.graph_objs._figure.Figure: A plot showing the stock price data and the Bollinger Bands.
    """
    # Fetch the stock data
    data = get_stock_data(ticker, start_date, end_date)

    # Define the window size and number of standard deviations for the Bollinger Bands
    window_size = 20
    num_of_std = 2

    # Calculate the rolling mean of the 'Close' prices over the specified window size
    rolling_mean = data['Close'].rolling(window=window_size).mean()

    # Calculate the rolling standard deviation of the 'Close' prices over the specified window size
    rolling_std = data['Close'].rolling(window=window_size).std()

    # Calculate the upper Bollinger Band as the rolling mean plus a number of standard deviations
    upper_band = rolling_mean + (rolling_std * num_of_std)

    # Calculate the lower Bollinger Band as the rolling mean minus a number of standard deviations
    lower_band = rolling_mean - (rolling_std * num_of_std)

    # Create a candlestick chart with the 'Open', 'High', 'Low', and 'Close' prices
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    # Add the upper Bollinger Band to the chart as a red line
    fig.add_trace(go.Scatter(x=data.index, y=upper_band, name="Upper Band", line=dict(color='red')))

    # Add the rolling mean to the chart as a blue line
    fig.add_trace(go.Scatter(x=data.index, y=rolling_mean, name="Rolling Mean", line=dict(color='blue')))

    # Add the lower Bollinger Band to the chart as a green line
    fig.add_trace(go.Scatter(x=data.index, y=lower_band, name="Lower Band", line=dict(color='green')))

    # Set the title of the chart to 'Bollinger Bands for {ticker}'
    fig.update_layout(title='Bollinger Bands for {}'.format(ticker))

    # Display the chart
    fig.show()

# This function plots a specified economic indicator for a list of countries over a given time period.
def countries_indicator(indicator, countries, start_date, end_date):
    """
    Fetches and plots an economic indicator for a list of countries.

    Parameters:
    indicator (str): The World Bank indicator code of interest.
    countries (list): A list of country ISO codes.
    start_date (str): The start year for the data in YYYY format.
    end_date (str): The end year for the data in YYYY format.
    """
    fig = go.Figure()

    indicator_name = None

    # Loop over each country, fetch the data, and add a line to the plot for that country
    for country in countries:
        df = get_world_bank_data(indicator, country, start_date, end_date)
        if not df.empty:
            fig.add_trace(go.Scatter(x=df.index, y=df['value'], mode='lines', name=country))
            # Get the indicator name from the first row of the DataFrame
            if indicator_name is None:
                indicator_name = df['indicator'].iloc[0]

    # Update the layout of the plot with the title and axis labels
    if indicator_name is not None:
        fig.update_layout(title=f'{indicator_name} from {start_date} to {end_date}', xaxis_title='Year', yaxis_title=indicator_name)
    else:
        fig.update_layout(title=f'{indicator} from {start_date} to {end_date}', xaxis_title='Year', yaxis_title=indicator)

    # Display the plot
    fig.show()

# This function is used to analyze and plot the correlation between daily returns of different stocks over a specified time period.
def daily_returns(tickers, start_date, end_date):
    """
    This function fetches stock data for a list of tickers over a specified time period, calculates daily returns and plots them.
    
    Parameters:
    tickers (list): A list of ticker symbols of the stocks.
    start_date (str): The start date for the data. The date should be in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data. The date should be in 'YYYY-MM-DD' format.

    Returns:
    plotly.graph_objs._figure.Figure: A plot showing the daily returns of the stocks.
    """
    # Get the stock data
    stock_data = pd.DataFrame()
    for ticker in tickers:
        data = get_stock_data(ticker, start_date, end_date)
        stock_data[ticker] = data['Close']
    
    # Calculate daily returns in percentage
    returns = stock_data.pct_change() * 100
    
    # Create a figure
    fig = go.Figure()

    # Define a list of colors for the lines
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

    # Add traces for each symbol
    for i, symbol in enumerate(tickers):
        fig.add_trace(go.Scatter(x=returns.index, y=returns[symbol], name=symbol,
                                 line=dict(width=1, color=colors[i % len(colors)]), opacity=0.7))

    # Set the title and axis labels
    fig.update_layout(title='Daily Returns of Stocks',
                      xaxis_title='Date',
                      yaxis_title='Daily Return (%)',
                      autosize=False,
                      width=1000,
                      height=500,
                      margin=dict(l=50, r=50, b=100, t=100, pad=4),
                      paper_bgcolor="white",
                      plot_bgcolor='white')

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # Show the figure
    fig.show()

# This function plots the stock price prediction for a given ticker using the ARIMA model.
def arima_prediction(ticker, start_date, end_date):
    """
    This function uses the ARIMA model to predict the stock price for a given ticker. It first fetches the stock data for the given ticker and date range. 
    Then, it calculates various technical indicators such as moving averages, Bollinger Bands, RSI, and MACD. 
    An ARIMA model is fit to the closing prices of the stock, and a prediction is made for the next day. 
    The function then plots the closing prices, moving averages, Bollinger Bands, RSI, and MACD along with the predicted price.

    Parameters:
    ticker (str): The ticker symbol of the stock.
    start_date (str): The start date for fetching the stock data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching the stock data in 'YYYY-MM-DD' format.

    Returns:
    The function directly plots the data using Plotly.
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Get the stock data
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Calculate moving averages
    stock_data['MA10'] = stock_data['Close'].rolling(10).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(50).mean()
    
    # Calculate Bollinger Bands
    stock_data['20 Day MA'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['20 Day STD'] = stock_data['Close'].rolling(window=20).std()
    stock_data['Upper Band'] = stock_data['20 Day MA'] + (stock_data['20 Day STD'] * 2)
    stock_data['Lower Band'] = stock_data['20 Day MA'] - (stock_data['20 Day STD'] * 2)
    
    # Calculate RSI
    delta = stock_data['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=14).mean()
    average_loss = abs(down.rolling(window=14).mean())
    rs = average_gain / average_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    stock_data['26 EMA'] = stock_data['Close'].ewm(span=26).mean()
    stock_data['12 EMA'] = stock_data['Close'].ewm(span=12).mean()
    stock_data['MACD'] = stock_data['12 EMA'] - stock_data['26 EMA']
    stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9).mean()
    
    # Fit the ARIMA model
    arima_order = (5,1,0)  # Hardcoded ARIMA order
    model = ARIMA(stock_data['Close'], order=arima_order)
    model_fit = model.fit()
    
    # Make the prediction
    forecast_result = model_fit.forecast(steps=1)
    prediction = forecast_result.iloc[0]

    # Create a new DataFrame for the prediction
    prediction_data = pd.DataFrame({'Close': [prediction]}, index=[pd.to_datetime(end_date) + pd.DateOffset(days=1)])

    # Concatenate the historical and prediction data
    full_data = pd.concat([stock_data, prediction_data])

    # Create subplots
    fig = make_subplots(rows=3, cols=1)

    # Add traces
    fig.add_trace(go.Scatter(x=full_data.index, y=full_data['Close'], mode='lines', name='Close Price', line=dict(color='darkblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA10'], mode='lines', name='10-day MA', line=dict(color='darkred')), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='50-day MA', line=dict(color='darkgreen')), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper Band'], mode='lines', name='Upper Bollinger Band', line=dict(color='indigo')), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower Band'], mode='lines', name='Lower Bollinger Band', line=dict(color='darkorange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=prediction_data.index, y=prediction_data['Close'], mode='markers', name='Predicted Price', marker=dict(color='red', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='teal')), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='darkviolet')), row=3, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal Line'], mode='lines', name='Signal Line', line=dict(color='gold')), row=3, col=1)

    # Update x-axis properties
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    # Update y-axis properties
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    # Update title and size
    fig.update_layout(height=800, width=1000, title_text=f"{ticker} Stock Price Prediction: {prediction:.2f}$")

    fig.show()

# This function plots the cumulative returns of a portfolio
def cum_returns(tickers, start_date, end_date):
    """
    This function calculates the overall return of a portfolio and plots it.

    Parameters:
    tickers (list): A list of ticker symbols for the stocks in the portfolio.
    start_date (str): The start date for the period over which to calculate returns, in 'YYYY-MM-DD' format.
    end_date (str): The end date for the period over which to calculate returns, in 'YYYY-MM-DD' format.

    Returns:
    The function plots the cumlative returns of the portfolio.
    """
    # Create an empty DataFrame to store the return data for each stock
    portfolio_return = pd.DataFrame()

    # Loop over each ticker symbol in the list
    for ticker in tickers:
        # Fetch the stock data for the given ticker and date range
        stock_data = get_stock_data(ticker, start_date, end_date)

        # Calculate the daily return for the stock by finding the percentage change in the closing price
        stock_data['Return'] = stock_data['Close'].pct_change()

        # If this is the first stock we're processing, copy its return data into the portfolio_return DataFrame
        if portfolio_return.empty:
            portfolio_return = stock_data[['Return']].copy()
            # Rename the column to the ticker symbol
            portfolio_return.columns = [ticker]
        else:
            # If this is not the first stock, join its return data to the existing portfolio_return DataFrame
            portfolio_return = portfolio_return.join(stock_data['Return'], how='outer', rsuffix=ticker)

    # Calculate the average daily return across all stocks in the portfolio
    portfolio_return['Average'] = portfolio_return.mean(axis=1)

    # Calculate the cumulative return of the portfolio by taking the cumulative product of (1 + the average daily return)
    # This represents the total return of the portfolio over the specified time period
    portfolio_return['Cumulative'] = (1 + portfolio_return['Average']).cumprod()

    # Convert the cumulative return to a percentage for easier interpretation
    portfolio_return['Cumulative'] = portfolio_return['Cumulative'] * 100

    # Create a Plotly figure to visualize the cumulative return
    fig = go.Figure(data=go.Scatter(x=portfolio_return.index, y=portfolio_return['Cumulative']))

    # Set the title and axis labels for the plot
    fig.update_layout(title='Portfolio Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Return (%)')

    # Display the plot
    fig.show()

# This function plots the annual returns of a portfolio
def annual_returns(tickers, start_year, end_year):
    """
    Plot the annual return of a portfolio over a range of years.

    Parameters:
    tickers (list of str): Ticker symbols of the stocks.
    start_year (int): Start year of the period (format: YYYY)
    end_year (int): End year of the period (format: YYYY)

    Returns:
    The function directly plots the annual returns of a portfolio using Plotly.
    """
    # Define a nested function to calculate the annual return of a single stock
    def calculate_annual_return(ticker, start_date, end_date):
        """
        Calculate the annual return of a stock.

        Parameters:
        ticker (str): Ticker symbol of the stock.
        start_date (str): Start date of the period.
        end_date (str): End date of the period.

        Returns:
        float: Annual return of the stock.
        """
        # Fetch the stock data for the given ticker and date range
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close']

        # Calculate the annual return by comparing the start and end prices and adjusting for the number of years
        start_price = stock_data.iloc[0]
        end_price = stock_data.iloc[-1]
        years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
        annual_return = ((end_price / start_price) ** (1 / years)) - 1

        # If the return value is a Series or DataFrame (which can happen if the stock data is empty), take the first value
        if isinstance(annual_return, (pd.Series, pd.DataFrame)):
            annual_return = annual_return.iloc[0]

        return annual_return

    # Calculate the annual returns for each year in the period for each ticker
    years = np.arange(start_year, end_year + 1)
    annual_returns = {ticker: [calculate_annual_return(ticker, str(year) + '-01-01', str(year) + '-12-31') for year in years] for ticker in tickers}

    # Create a bar chart for the annual returns of each stock
    traces = [go.Bar(x=years, y=returns, name=ticker) for ticker, returns in annual_returns.items()]

    # Define the layout for the plot
    layout = go.Layout(title=f'Annual Returns of Portfolio from {start_year} to {end_year}',
                       xaxis=dict(title='Year'),
                       yaxis=dict(title='Return (%)', tickformat=".2%"),  # Format the y-axis as a percentage
                       showlegend=True)

    # Create the figure and add the traces
    fig = go.Figure(data=traces, layout=layout)

    # Display the figure
    fig.show()

# This function plots the rolling volatility of a stock
def rolling_volatility(ticker, start_date, end_date, window):
    """
    Plot the rolling volatility of a stock.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start_date (str): Start date of the period (format: "YYYY-MM-DD").
    end_date (str): End date of the period (format: "YYY-MM-DD").
    window (int): Rolling window size for volatility calculation.
    """
    # Fetch the stock data for the given ticker and date range
    data = get_stock_data(ticker, start_date, end_date)

    # Calculate the rolling volatility by taking the standard deviation of the percentage change in the 'Adj Close' price
    rolling_volatility = data['Adj Close'].pct_change().rolling(window).std()

    # Calculate the moving average of the rolling volatility
    moving_average_volatility = rolling_volatility.rolling(window).mean()

    # Initialize a plotly figure
    fig = go.Figure()

    # Add a trace for the rolling volatility
    fig.add_trace(go.Scatter(
        x=rolling_volatility.index,
        y=rolling_volatility.values,
        mode='lines',
        name=f'{ticker} Rolling Volatility',
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Volatility: %{y}',
        fill='tozeroy'  # Fill the area under the line
    ))

    # Add a trace for the moving average volatility
    fig.add_trace(go.Scatter(
        x=moving_average_volatility.index,
        y=moving_average_volatility.values,
        mode='lines',
        name=f'{ticker} Moving Average Volatility',
        line=dict(color='red'),
        hovertemplate='Date: %{x}<br>Average Volatility: %{y}'
    ))

    # Update the layout of the figure
    fig.update_layout(
        title=f'{ticker} Stock Rolling Volatility from {start_date} to {end_date}',
        xaxis_title='Date',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),  # Add a range slider to the x-axis
            type="date"
        ),
        yaxis_title='Rolling Volatility',
        hovermode='x'  # Show hover information for the x-axis value
    )

    # Display the figure
    fig.show()

# This function plots the Moving Average Convergence Divergence (MACD) for a stock
def macd(ticker, start_date, end_date, short_window, long_window):
    """
    Plot the Moving Average Convergence Divergence (MACD) for a stock.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start_date (str): Start date of the period (format: "YYYY-MM-DD").
    end_date (str): End date of the period. (format: "YYYY-MM-DD").
    short_window (int): Short window size for EMA calculation.
    long_window (int): Long window size for EMA calculation.
    """
    # Fetch the stock data for the given ticker and date range
    data = get_stock_data(ticker, start_date, end_date)

    # Calculate the short-term Exponential Moving Average (EMA)
    data['short_ema'] = data['Adj Close'].ewm(span=short_window, adjust=False).mean()

    # Calculate the long-term EMA
    data['long_ema'] = data['Adj Close'].ewm(span=long_window, adjust=False).mean()

    # Calculate the MACD line as the difference between short-term and long-term EMA
    data['macd_line'] = data['short_ema'] - data['long_ema']

    # Calculate the signal line as the 9-day EMA of the MACD line
    data['signal_line'] = data['macd_line'].ewm(span=9, adjust=False).mean()

    # Calculate the MACD difference (also known as MACD histogram) as the difference between MACD line and signal line
    data['macd_difference'] = data['macd_line'] - data['signal_line']

    # Initialize a plotly figure with two subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=("MACD Line and Signal Line", "MACD Difference"))

    # Add a trace for the MACD line to the first subplot
    fig.add_trace(go.Scatter(x=data.index, y=data['macd_line'], name='MACD Line (Short EMA - Long EMA)', line=dict(color='blue')), row=1, col=1)

    # Add a trace for the signal line to the first subplot
    fig.add_trace(go.Scatter(x=data.index, y=data['signal_line'], name='Signal Line (9-day EMA of MACD Line)', line=dict(color='red')), row=1, col=1)

    # Add a bar chart for the MACD difference to the second subplot
    fig.add_trace(go.Bar(x=data.index, y=data['macd_difference'], name='MACD Difference (MACD Line - Signal Line)', marker_color='purple'), row=2, col=1)

    # Add a zero line to the second subplot
    fig.add_trace(go.Scatter(x=data.index, y=[0]*len(data.index), name='Zero Line', line=dict(color='gray', dash='dash')), row=2, col=1)

    # Update the layout of the figure
    fig.update_layout(
        title=f'Moving Average Convergence Divergence (MACD) for {ticker} from {start_date} to {end_date}',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x'  # Show hover information for the x-axis value
    )

    # Display the figure
    fig.show()

# This function plots the Relative Strength Index (RSI) for a stock
def rsi(ticker, start_date, end_date, window_size):
    """
    Plot the Relative Strength Index (RSI) for a stock.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start_date (str): Start date of the period (format: "YYYY-MM-DD").
    end_date (str): End date of the period (format: "YYYY-MM-DD").
    window_size (int): Window size for RSI calculation.
    """
    # Fetch the stock data for the given ticker and date range
    data = get_stock_data(ticker, start_date, end_date)

    # Calculate the difference between consecutive adjusted closing prices
    delta = data['Adj Close'].diff()

    # Create two copies of the delta series for gain and loss calculation
    up, down = delta.copy(), delta.copy()

    # Replace negative values with 0 in the 'up' series
    up[up < 0] = 0

    # Replace positive values with 0 in the 'down' series
    down[down > 0] = 0

    # Calculate the average gain over the window size
    average_gain = up.rolling(window=window_size).mean()

    # Calculate the average loss over the window size
    average_loss = abs(down.rolling(window=window_size).mean())

    # Calculate the Relative Strength (RS) as the ratio of average gain to average loss
    rs = average_gain / average_loss

    # Calculate the RSI as 100 - (100 / (1 + RS))
    data['RSI'] = 100 - (100 / (1 + rs))

    # Initialize a plotly figure
    fig = go.Figure()

    # Add a trace for the RSI to the figure
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='blue')))

    # Add a trace for the overbought line (RSI = 70) to the figure
    fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data.index), name='Overbought Line (70)', line=dict(color='red', dash='dash')))

    # Add a trace for the oversold line (RSI = 30) to the figure
    fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data.index), name='Oversold Line (30)', line=dict(color='green', dash='dash')))

    # Update the layout of the figure
    fig.update_layout(
        title=f'Relative Strength Index (RSI) for {ticker} from {start_date} to {end_date}',
        xaxis_title='Date',
        yaxis_title='RSI',
        hovermode='x'  # Show hover information for the x-axis value
    )

    # Display the figure
    fig.show()

# This function plots the Volume Weighted Average Price (VWAP) for a stock
def vwap(ticker, start_date, end_date):
    """
    Plot the Volume Weighted Average Price (VWAP) for a stock.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start_date (str): Start date of the period (format: "YYYY-MM-DD").
    end_date (str): End date of the period (format: "YYYY-MM-DD").
    """
    # Fetch the stock data for the given ticker and date range
    data = get_stock_data(ticker, start_date, end_date)

    # Calculate the VWAP as the cumulative sum of the product of volume and average price (high + low / 2) divided by the cumulative sum of volume
    data['VWAP'] = np.cumsum(data['Volume'] * (data['High'] + data['Low']) / 2) / np.cumsum(data['Volume'])

    # Initialize a plotly figure
    fig = go.Figure()

    # Add a trace for the adjusted closing price to the figure
    fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], name='Adj Close', line=dict(color='blue')))

    # Add a trace for the VWAP to the figure
    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name='VWAP', line=dict(color='red')))

    # Update the layout of the figure
    fig.update_layout(
        title=f'Volume Weighted Average Price (VWAP) for {ticker} from {start_date} to {end_date}',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x'  # Show hover information for the x-axis value
    )

    # Display the figure
    fig.show()

# This function predicts stock prices using linear regression and plots the results
def lm_prediction(ticker, start_date, end_date, window_size):
    """
    Predict stock prices using linear regression and plot the results.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start_date (str): Start date of the period (format: "YYYY-MM-DD").
    end_date (str): End date of the period (format: "YYYY-MM-DD").
    window_size (int): Window size for the rolling linear regression model.
    """
    # Fetch the stock data for the given ticker and date range
    data = get_stock_data(ticker, start_date, end_date)

    # Initialize a linear regression model
    model = LinearRegression()
    
    # Create a rolling window for the linear regression model
    predictions = []
    for i in range(window_size, len(data)):
        # Define the training data for the model
        x_train = np.arange(window_size).reshape(-1, 1)  # Independent variable (time)
        y_train = data['Adj Close'].iloc[i-window_size:i].values.reshape(-1, 1)  # Dependent variable (adjusted close price)

        # Train the model
        model.fit(x_train, y_train)

        # Predict the price for the next day and append it to the predictions list
        predictions.append(model.predict([[window_size]])[0][0])
    
    # Append NaNs at beginning of predictions to match length of original data
    predictions = [np.nan]*window_size + predictions

    # Add the predictions to the data DataFrame
    data['Predictions'] = predictions

    # Initialize a plotly figure
    fig = go.Figure()

    # Add a trace for the actual prices to the figure
    fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Actual Price'))

    # Add a trace for the predicted prices to the figure
    fig.add_trace(go.Scatter(x=data.index, y=data['Predictions'], mode='lines', name='Predicted Price'))

    # Update the layout of the figure
    fig.update_layout(
        title=f'Linear Regression Prediction for {ticker} from {start_date} to {end_date}',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x'  # Show hover information for the x-axis value
    )

    # Display the figure
    fig.show()

# This function plots Fibonacci retracement levels for a stock
def fibonacci_retracement(ticker, start_date, end_date):
    """
    Plot Fibonacci retracement levels for a stock.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start_date (str): Start date of the period (format: "YYYY-MM-DD").
    end_date (str): End date of the period. (format: "YYYY-MM-DD").
    """
    # Fetch the stock data for the given ticker and date range
    data = get_stock_data(ticker, start_date, end_date)

    # Find the maximum and minimum closing prices
    max_price = data['Close'].max()
    min_price = data['Close'].min()

    # Calculate the difference between the maximum and minimum prices
    diff = max_price - min_price

    # Define the Fibonacci retracement levels
    levels = [0.0, 0.236, 0.382, 0.618, 1.0]

    # Calculate the price at each Fibonacci retracement level
    prices = [min_price + diff*level for level in levels]

    # Define the colors and labels for the Fibonacci retracement levels
    colors = ["Red", "Orange", "Green", "Blue", "Purple"]
    labels = ['0%', '23.6%', '38.2%', '61.8%', '100%']

    # Initialize a plotly figure
    fig = go.Figure()

    # Add a trace for the closing prices to the figure
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

    # Add a line and a marker for each Fibonacci retracement level to the figure
    for i, price in enumerate(prices):
        fig.add_shape(type="line", x0=data.index[0], y0=price, x1=data.index[-1], y1=price, line=dict(color=colors[i], width=1))
        fig.add_trace(go.Scatter(x=[data.index[-1]], y=[price], mode='markers', marker=dict(color=colors[i]), name=f'Fibonacci {labels[i]}', hovertemplate=f'Fibonacci {labels[i]}: {price}'))

    # Update the layout of the figure
    fig.update_layout(
        title=f'Fibonacci Retracement Levels for {ticker} from {start_date} to {end_date}',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x',  # Show hover information for the x-axis value
        xaxis_showgrid=True,  # Show gridlines for the x-axis
        yaxis_showgrid=True,  # Show gridlines for the y-axis
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Display the figure
    fig.show()

# This function plots the Ichimoku Cloud for a stock
def ichimoku_cloud(ticker, start_date, end_date):
    """
    Plot the Ichimoku Cloud for a stock.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start_date (str): Start date of the period (format: "YYYY-MM-DD").
    end_date (str): End date of the period (format: "YYYY-MM-DD").
    """
    # Fetch the stock data for the given ticker and date range
    data = get_stock_data(ticker, start_date, end_date)

    # Extract the high, close, and low prices from the data
    high_prices = data['High']
    close_prices = data['Close']
    low_prices = data['Low']

    # Calculate the Conversion Line (Tenkan Sen) as the average of the 9-period high and low
    nine_period_high = high_prices.rolling(window=9).max()
    nine_period_low = low_prices.rolling(window=9).min()
    data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Calculate the Base Line (Kijun Sen) as the average of the 26-period high and low
    period26_high = high_prices.rolling(window=26).max()
    period26_low = low_prices.rolling(window=26).min()
    data['kijun_sen'] = (period26_high + period26_low) / 2

    # Calculate Leading Span A (Senkou Span A) as the average of the Conversion Line and the Base Line, shifted 26 periods ahead
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)

    # Calculate Leading Span B (Senkou Span B) as the average of the 52-period high and low, shifted 26 periods ahead
    period52_high = high_prices.rolling(window=52).max()
    period52_low = low_prices.rolling(window=52).min()
    data['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    # Calculate the Lagging Span (Chikou Span) as the closing price, shifted 26 periods behind
    data['chikou_span'] = close_prices.shift(-26)

    # Initialize a plotly figure
    fig = go.Figure()

    # Add traces for the closing price and the Ichimoku Cloud indicators to the figure
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['tenkan_sen'], mode='lines', name='Conversion Line (Tenkan Sen)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['kijun_sen'], mode='lines', name='Baseline (Kijun Sen)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['senkou_span_a'], mode='lines', name='Leading Span A (Senkou Span A)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['senkou_span_b'], mode='lines', name='Leading Span B (Senkou Span B)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['chikou_span'], mode='lines', name='Lagging Span (Chikou Span)'))

    # Update the layout of the figure
    fig.update_layout(
        title=f'Ichimoku Cloud for {ticker} from {start_date} to {end_date}',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x',  # Show hover information for the x-axis value
        xaxis_showgrid=True,  # Show gridlines for the x-axis
        yaxis_showgrid=True,  # Show gridlines for the y-axis
    )

    # Display the figure
    fig.show()

# This function plots a comparison of GDP growth for multiple countries
def gdp_growth(countries, start_date, end_date):
    """
    This function plots the GDP growth comparison for the given countries over a specified time period.

    Parameters:
    countries (list): List of country ISO codes.
    start_date (str): Start date of the period (format: "YYYY").
    end_date (str): End date of the period (format: "YYYY").
    """
    # Initialize a plotly figure
    fig = go.Figure()

    # Loop over each country
    for country in countries:
        # Fetch the GDP growth data for the country from the World Bank
        df = get_world_bank_data('NY.GDP.MKTP.KD.ZG', country, start_date, end_date)

        # Round the GDP growth values to 2 decimal places
        df['value'] = df['value'].round(2)

        # Add a trace for the country's GDP growth to the figure
        fig.add_trace(go.Scatter(x=df.index, y=df['value'], mode='lines', name=f'{country} GDP Growth'))

    # Update the layout of the figure
    fig.update_layout(
        title=f'GDP Growth Comparison from {start_date} to {end_date}',  # Set the title of the figure
        xaxis_title='Year',  # Set the title of the x-axis
        yaxis_title='GDP Growth (Annual %)',  # Set the title of the y-axis
        hovermode='x',  # Show hover information for the x-axis value
        xaxis_showgrid=True,  # Show gridlines for the x-axis
        yaxis_showgrid=True,  # Show gridlines for the y-axis
    )

    # Display the figure
    fig.show()

# This function plots an economic analysis for multiple countries
def analyse_economy(countries, start_date, end_date):
    """
    This function plots the economic analysis of the given countries over a specified time period.

    Parameters:
    countries (list): List of country ISO codes.
    start_date (str): Start date of the period (format: "YYYY").
    end_date (str): End date of the period (format: "YYYY").
    """
    # Initialize a subplot figure with 4 rows for different economic indicators
    fig = sp.make_subplots(rows=4, cols=1, subplot_titles=("GDP (Current US$)", "Unemployment Rate (%)", "Inflation Rate (%)", "Debt-to-GDP Ratio (%)"))

    # Loop over each country
    for country in countries:
        # Fetch the GDP data for the country from the World Bank
        df_gdp = get_world_bank_data('NY.GDP.MKTP.CD', country, start_date, end_date)
        # Add a trace for the country's GDP to the figure
        fig.add_trace(go.Scatter(x=df_gdp.index, y=df_gdp['value'], mode='lines', name=f'{country} GDP'), row=1, col=1)

        # Fetch the Unemployment Rate data for the country from the World Bank
        df_unemployment = get_world_bank_data('SL.UEM.TOTL.ZS', country, start_date, end_date)
        # Add a trace for the country's Unemployment Rate to the figure
        fig.add_trace(go.Scatter(x=df_unemployment.index, y=df_unemployment['value'], mode='lines', name=f'{country} Unemployment Rate'), row=2, col=1)

        # Fetch the Inflation Rate data for the country from the World Bank
        df_inflation = get_world_bank_data('FP.CPI.TOTL.ZG', country, start_date, end_date)
        # Add a trace for the country's Inflation Rate to the figure
        fig.add_trace(go.Scatter(x=df_inflation.index, y=df_inflation['value'], mode='lines', name=f'{country} Inflation Rate'), row=3, col=1)

        # Fetch the Debt-to-GDP Ratio data for the country from the World Bank
        df_debt = get_world_bank_data('GC.DOD.TOTL.GD.ZS', country, start_date, end_date)
        # Add a trace for the country's Debt-to-GDP Ratio to the figure
        fig.add_trace(go.Scatter(x=df_debt.index, y=df_debt['value'], mode='lines', name=f'{country} Debt-to-GDP'), row=4, col=1)

    # Update the layout of the figure
    fig.update_layout(height=800, title=f'Economic Analysis of {", ".join(countries)} from {start_date} to {end_date}', hovermode='x')
    # Set the title of the x-axis for each subplot
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_xaxes(title_text="Year", row=4, col=1)
    # Set the title of the y-axis for each subplot
    fig.update_yaxes(title_text="Current US$", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="%", row=4, col=1)

    # Display the figure
    fig.show()